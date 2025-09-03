#!/usr/bin/env python
from itertools import chain
from joblib import Parallel, delayed
from lightning.pytorch.tuner import Tuner
import argparse
from copy import deepcopy
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
from numpy import dtype
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from phyloDNN import seq_utils as sq
from phyloDNN import utils as u
from phyloDNN.bionj import BioNJ
from phyloDNN.losses import (
    L21Loss,
    LSELoss,
    LogDetLoss,
    RelativeLoss,
    SoftMaxLoss,
    VonNeumannLoss,
)
from phyloDNN.models.graph_utils import make_indices

from phyloDNN.small_nets import (
    BourgainNet,
    GTRNet,
    JCNetExact,
    K2PNetExact,
    SmallNet,
    JCNet,
    K2PNet,
    HammingExact,
)

torch.set_float32_matmul_precision("medium")


class LightningAgent(pl.LightningModule):

    def __init__(
        self,
        rootdir: Path,
        seqdir: str,
        num_workers: int = 8,
        network: str = "small",
        max_length: int = 500,
        valdir: Path | None = None,
        verbose=False,
        regularizer=None,
        transform=None,
        eta=0.2,
        pin_memory: bool = True,
        compile=False,
        radial=False,
        loss="mse",
        keep_invariant: bool = True,
        alpha: float = 1.0,
        treefile="trees.tsv",
        batch_size: int | None = None,
        learning_rate: float = 0.01,
        transfer=False,
        ml_dist: str = None,
        opt: str = "adam",
        polyak: float | None = 0.995,
        val_alg: str = "nj",
        **kwargs,
    ):
        """Initialize LightningAgent
        Args:
            rootdir (Path): dir to look for raw/processed data
            valdir (Path): dir to look for validation data
            network (str): network architecture
            max_length (int): max length of alignment for train/val sets
            num_workers (int): number of workers for dataloader
            verbose (bool): whether to print progress
            regularizer (str): regularizer to use
            transform (callable): transform to apply to data
            eta (float): weight for regularizer
            preprocessor (callable): preprocessor to apply to data
            pin_memory (bool): whether to pin memory
            seqdir (str): directory to look for sequences
            compile (bool): whether to compile the model
            radial (bool): whether to use radial regularizer
            loss (str): loss function to use
            keep_invariant (bool): whether to keep invariant sequences
            alpha (float): alpha parameter for loss function
        Args:
            seqdir (Path): dir to look for phylip seqs
            epochs (int): max epochs
        """

        super().__init__()
        self.save_hyperparameters()
        # TODO: add option for EdgeBlockNet; need to reshape batch_size*ntaxa and generate edge_index a la QFunction
        self.is_graph_net = False
        self.pin_memory = pin_memory
        self.opt = opt
        if network == "jcnet":
            self.policy_function = JCNet(**kwargs)
        elif network == "gtrnet":
            self.policy_function = GTRNet(**kwargs)
        elif network == "bourgain":
            self.policy_function = BourgainNet(**kwargs)
        elif network == "k2pnet" or network == "hkynet":
            self.policy_function = K2PNet(**kwargs)
        elif network == "k2pnet_exact":
            self.policy_function = K2PNetExact()
        elif network == "jcnet_exact":
            # Jukes-Cantor exact model has a single parameter
            self.policy_function = JCNetExact(**kwargs)
        elif network == "hamming":
            self.policy_function = HammingExact()
        else:
            self.policy_function = u.make_network(network, **kwargs)
            # self.policy_function = SmallNet(**kwargs)

        self.matrix_type = kwargs.get("format", "distance")
        if self.matrix_type not in ("distance", "covariance"):
            raise ValueError("matrix type must be distance or covariance")
        # TODO: add transfer net option.
        self.transform = transform
        self.treefile = treefile
        self.batches = 0
        self.rootdir = rootdir
        self.seqdir = seqdir
        self.valdir = valdir
        self.max_train_length = max_length
        self.workers = num_workers
        self.val_alg = val_alg
        self.verbose = verbose
        self.automatic_optimization = False
        self.regularizer = regularizer
        self.eta = eta

        if ml_dist is not None:

            if ml_dist == "jc":
                self.ml_dist_target = JCNetExact().eval()
            elif ml_dist == "k2p":
                self.ml_dist_target = K2PNetExact().eval()
            elif ml_dist == "hamming":
                self.ml_dist_target = HammingExact().eval()
            self.ml_dist_target.compile()

        if loss == "softmax":
            self.criterion = SoftMaxLoss(alpha=alpha)
        elif loss == "lse":
            self.criterion = LSELoss(alpha=alpha, ceil=True, l21=True)
        elif loss == "rel_mse":
            self.criterion = RelativeLoss(torch.nn.MSELoss)
        elif loss == "mse":
            self.criterion = torch.nn.MSELoss(reduction="mean")
        elif loss == "mae":
            self.criterion = torch.nn.L1Loss(reduction="mean")
        elif loss == "l21":
            self.criterion = L21Loss()
        else:
            raise ValueError(f"Unknown loss function {loss}")
        if regularizer is not None:
            if regularizer == "logdet":
                self.regularized_criterion = LogDetLoss(
                    format=self.matrix_type, radial=radial
                )
            elif regularizer == "von_neumann":
                self.regularized_criterion = VonNeumannLoss(format=self.matrix_type)
            self.regularized_criterion.compile()
        self.polyak = polyak

        self.criterion.compile()

        if polyak is not None:
            self.pi_targ = deepcopy(self.policy_function)

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            # for p in self.q_targ.parameters():
            #     p.requires_grad = False
            for p in self.pi_targ.parameters():
                p.requires_grad = False
        else:
            self.pi_targ = self.policy_function

        if compile:
            self.compile()

    def set_policy_function(self, policy_function):
        """Set the policy/target function explicitly"""
        self.policy_function = policy_function
        self.pi_targ = policy_function
        return self

    def compile(self):
        """Compile the model using torch.compile"""
        self.policy_function = torch.compile(
            self.policy_function,
            dynamic=True,
            fullgraph=False,
            # backend="nvprims_nvfuser",
            mode="reduce-overhead",
        )  # "max-autotune"
        if self.polyak is None:
            self.pi_targ = self.policy_function  # share
        else:
            self.pi_targ = torch.compile(
                self.pi_targ,
                dynamic=True,
                # backend="nvprims_nvfuser",
                fullgraph=False,
                mode="reduce-overhead",
            )

    def configure_optimizers(self):
        """Uses Adam with learning rate scheduler

        Args:
            pi_lr (float, optional): Initial lr of phyloformer is 0.00039. Defaults to .001.

        Returns:
            _type_: _description_
        """
        if self.opt == "sgd":
            optimizer = SGD(
                filter(lambda p: p.requires_grad, self.policy_function.parameters()),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                nesterov=True,
                fused=True,
            )
        elif self.opt == "adamw":
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.policy_function.parameters()),
                lr=torch.tensor(
                    self.hparams.learning_rate
                ),  # wrapping in tensor might be necessary for torch.compile
            )
        elif self.opt == "adam":
            optimizer = Adam(
                filter(lambda p: p.requires_grad, self.policy_function.parameters()),
                lr=self.hparams.learning_rate,
                foreach=True,
                fused=False,  # todo: try fused?
            )
        else:
            raise NotImplementedError(f"optimizer {self.opt} not implemented")
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,  # self.current_epoch,
            eta_min=1e-5,  # self.current_epoch,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            # "monitor": "val_drf",
        }

    def update_targets(self):
        # Finally, update target networks by polyak averaging.
        if self.polyak is None:
            raise ValueError("polyak is None, no target to update")
        polyak = self.polyak
        with torch.no_grad():
            for p, p_targ in zip(
                self.policy_function.parameters(), self.pi_targ.parameters()
            ):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data.to(p_targ.device))

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/batch_size": self.hparams.batch_size,
                "hp/lr": self.hparams.learning_rate,
            },
        )

    def training_step(self, data, batch_idx):
        """Training step for the model
        Args:
            data (torch.Tensor): input data
            batch_idx (int): batch index
        """
        alignment, *_, y = data

        if hasattr(self, "ml_dist_target"):
            y = self.ml_dist_target(alignment)

        y.to(self.device, non_blocking=True)
        if self.is_graph_net:
            alignment = self.make_batch(alignment)  # .to(self.device)

        ypred = self.policy_function(alignment)
        # with torch.autocast(enabled=False):
        if hasattr(self, "regularized_criterion"):
            if self.eta < 1:
                loss = self.criterion(ypred, y)
            else:
                loss = 0
            reg = self.regularized_criterion(ypred, y)
            self.log("loss/raw", loss, on_step=True)
            self.log(f"loss/{self.regularizer}", reg, on_step=True)
            loss = (1 - self.eta) * loss + self.eta * reg
        else:
            loss = self.criterion(ypred, y)
            self.log("loss", loss, on_step=True, prog_bar=True)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        # self.clip_gradients(optimizer, gradient_clip_val=5,
        #                     gradient_clip_algorithm="value")

        optimizer.step()
        scheduler.step(
            self.current_epoch + batch_idx / len(self.trainer.train_dataloader)
        )

        if self.polyak is not None:
            self.update_targets()

    def make_batch(self, x):
        batch_size, n_seqs = x.shape[:2]
        edges = (
            make_indices(batch_size, n_seqs).long().to(self.device, non_blocking=True)
        )
        batch_ix = (
            torch.arange(0, batch_size)
            .repeat_interleave(n_seqs)
            .long()
            .to(self.device, non_blocking=True)
        )
        batch = data.Batch(x=x.to(self.device), edge_index=edges, batch=batch_ix)
        return batch

        # for ytrue_batch, ypred_batch in zip(y, ypred):
        #     true_tree = sq.njtree(ytrue_batch)
        #     pred_tree = sq.njtree(ypred_batch, true_tree.taxon_namespace)
        #     d += calculate_dRF(pred_tree, true_tree)
        #     n += 1
        # return d / n

    def _acc(self, x: torch.Tensor, y: torch.Tensor, mode: str = "val") -> float:
        r"""compute d_RF.  applies ReLU to ensure distances are > 0

        Args:
            x (torch.Tensor): alignment
            y (torch.Tensor): pairwise distances (ALWAYS distances whether output of network is distance or covariance) between  :math:`i \neq j`.
            mode (str, optional): pi_targ or policy predictor. Defaults to 'val'.
            weighted (bool, optional): weighted robinson-foulds. Defaults to False.

        Returns:
            float:  :math:`d_{RF}(model(x), y)`
        """
        # TODO: inherit this for all agents
        if self.is_graph_net:
            x = self.make_batch(x)
        with torch.no_grad():
            if mode == "val" and self.polyak is not None:
                ypred = self.pi_targ(x).relu()
            else:
                ypred = self.policy_function(x).relu()

            try:
                ypred = ypred.reshape(y.shape).cpu()
            except RuntimeError:
                pass
        self.log(f"{mode}/loss", self.criterion(ypred, y.to(ypred.device)))

        d_mean, d_std = sq.d_RF(
            y,
            ypred,
            workers=self.workers,
            matrix_type=self.matrix_type,
            algorithm=self.val_alg,
        )

        if mode == "val" and self.regularizer == "logdet":
            # don't need squareform for building njtrees, but do for checking matrices
            ypred = u.squareform(ypred, add_diagonal=self.matrix_type != "covariance")

            # self.log('Y_cond', torch.linalg.cond(y).mean())
            try:
                c, L = self.regularized_criterion.conditioning(ypred)
                self.log("Ypred/cond", c)

                # L = torch.linalg.eigvalsh(y)
                # self.log('Y_pos_eig', L.relu().count_nonzero(1).float().mean())

                self.log("Ypred/pos_eig", L.relu().count_nonzero(1).float().mean())
            except torch._C._LinAlgError as e:
                print(e)

        self.log(f"{mode}/drf", d_mean)
        self.log(f"{mode}/drf_std", d_std)

        return d_mean

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, ids, filenames, *_ = batch
        if x.shape[0] == 1:
            ids, filenames = [ids], [filenames]
        return filenames, ids, self.pi_targ(x).relu()

    def validation_step(self, batch, batch_idx):
        x, *_, dists = batch
        if not isinstance(dists, torch.Tensor):
            raise ValueError("no distances in validation batch")
        acc = self._acc(x, dists, "val")
        # if not (batch_idx + 1) % 2000:
        #     pi_sched = self.lr_schedulers()
        #     pi_sched.step(acc)
        return acc

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """

        When both batch_size and batch_sampler are None (default value for batch_sampler is already None),
            automatic batching is disabled. Each sample obtained from the dataset is processed
             with the function passed `as the collate_fn argument.

        When automatic batching is disabled, the default collate_fn simply converts NumPy arrays into
            PyTorch Tensors, and keeps everything else untouched.

        pin_memory may speed up training, but need to ensure there is still enough memory to load validation data

        Returns:
            torch.utils.data.DataLoader: torch dataloader
        """
        if self.rootdir is None or not self.rootdir.exists():
            raise ValueError("no training data directory specified")
        return self.dataloader(
            self.rootdir,
            num_workers=self.workers,
            max_length=self.max_train_length,
            seqdir=self.seqdir,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Batch size by default is 1/8 of training batch size (assume we're testing on larger alignments).

        Returns:
            torch.utils.data.DataLoader: torch dataloader
        """
        torch.cuda.empty_cache()
        return self.dataloader(
            self.valdir,
            num_workers=self.workers and self.workers // 2,
            max_length=self.max_train_length,
            seqdir=self.seqdir,
            batch_size=self.hparams.batch_size // 8 or 2,
        )

    def dataloader(
        self,
        dirname: Path = None,
        seqdir: str = None,
        testdir: Path = None,
        num_workers: int | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        pin_memory: bool = False,
        phyloformer_style: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Create a dataloader for the given directory.
        If testdir is specified, use that directory and return taxa labels.
        If batch_size is not specified, use batch size from self.hparams.
        Returns:
            torch.utils.data.DataLoader: torch dataloader
        """
        if testdir is not None:
            dirname = testdir.parent
            seqdir = testdir.name

        if num_workers is None:
            num_workers = self.workers
        dataset = sq.AlignmentDataset(
            dirname,
            max_length=max_length,
            keep_invariant=self.hparams.keep_invariant,
            matrix_type="distance",
            transform=self.transform,
            seqdir=seqdir,
            true_distances=testdir is None,
            treefile=self.treefile,
            phyloformer_style=phyloformer_style,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=min(len(dataset), batch_size or self.hparams.batch_size),
            collate_fn=sq.collate_alignment_dataset,
            batch_sampler=None,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
            num_workers=num_workers,
            drop_last=False,
        )
        return dataloader


def main(args):
    import os

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        torch.cuda.device_count()  # print 1
    import warnings

    warnings.filterwarnings("ignore")
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(device)
    if args.config and args.config.exists():

        print(f"loading params from {args.config}")
        from json import load

        with open(args.config) as f:
            network_params = load(f)
    else:
        network_params = {}
    if "model" in network_params:
        network_params = network_params["model"]
    if (
        not args.predict
        and not args.summarize
        and not (args.rootdir.exists() and args.valdir.exists())
    ):
        raise ValueError("data dirs must exist")
    callbacks = [
        ModelCheckpoint(
            save_weights_only=False,  # True,
            monitor="val/drf",
            save_top_k=1,
            mode="min",
            filename="best_val_drf",
        ),
        ModelCheckpoint(
            save_weights_only=False,  # True,
            monitor="val/loss",
            save_top_k=1,
            mode="min",
            filename="best_val_loss",
        ),
        LearningRateMonitor(logging_interval="step", log_momentum=False),
    ]
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/drf",
                min_delta=1e-6,
                patience=20,
                verbose=False,
                mode="min",
            ),
        )
    if args.scale_y:
        if args.scale_y == 1:

            def scaler(t):
                h = t.get_farthest_leaf()[1]
                for n in t.iter_descendants():
                    n.dist /= h
                return t

        else:

            def scaler(t):
                for n in t.iter_descendants():
                    n.dist /= args.scale_y
                return t

    trainer = pl.Trainer(
        default_root_dir=args.checkpoint_dir,
        callbacks=callbacks,
        val_check_interval=args.check_interval,
        profiler=args.debug and "simple" or None,
        fast_dev_run=args.debug and 10,
        num_sanity_val_steps=2,
        # limit_train_batches=args.debug and 5 or None,
        limit_val_batches=(0.5 if args.batch_size and args.batch_size < 1000 else 1.0),
        check_val_every_n_epoch=(
            1 if args.check_interval is None or args.check_interval < 1 else None
        ),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=args.amp and "bf16" or "32-true",
        devices=1,
        # auto_scale_batch_size="binsearch",
        max_epochs=args.epochs,
        log_every_n_steps=20,
        enable_progress_bar=args.progressbar,
    )

    pretrained_filename = None

    agent_parameters = dict(
        batch_size=args.batch_size,
        rootdir=args.rootdir,
        seqdir=args.seqdir,
        valdir=args.valdir,
        transform=scaler if args.scale_y else None,
        num_workers=args.workers,
        pin_memory=args.pin_memory,  # torch.cuda.is_available(),
        compile=args.compile,
        polyak=args.polyak,
        val_alg=args.tree_alg,
        **network_params,
    )

    print('\n',args,'\n',agent_parameters,'\n')

    try:
        if args.phyloformer:
            raise ValueError("phyloformer is not a Lightning module")
        elif args.model_path is None:
            # Check whether pretrained model exists. If yes, load it and continue training
            args.model_path = u.get_last_checkpoint(args.checkpoint_dir)
        print(f"Found pretrained model, loading from {args.model_path}...")
        model = LightningAgent.load_from_checkpoint(args.model_path, **agent_parameters)
    except (ValueError, AttributeError):  # no checkpoint found
        if not args.phyloformer and args.config and not args.config.exists():
            raise IOError(
                "must specify either a configuration or a directory with a checkpointed model."
            )
        model = LightningAgent(**agent_parameters)
        if args.phyloformer:
            from pformer.model import Phyloformer

            ckpt = torch.load(args.model_path, map_location={"0": "cuda"})
            params = ckpt["hyper_parameters"]
            params["device"] = "cuda"
            model.set_policy_function(Phyloformer(**params))
            model.policy_function.load_state_dict(
                {
                    k.replace("model.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if k != "model.seq2pair"
                },
                strict=False,
            )
    if args.summarize:
        from torchinfo import summary

        print(args.model_path)
        summary(
            model.policy_function,
            torch.Size((4, 20, 1000)),
            dtypes=[torch.int32],
        )
        return
    if args.predict:
        import pandas as pd

        # from torchsummary import summary
        # print(summary(model.policy_function, (1, 60, 500)))

        dataloader = model.dataloader(
            testdir=args.predict,
            batch_size=args.batch_size if args.batch_size else 4,
            num_workers=args.workers,
            pin_memory=True,
            phyloformer_style=args.phyloformer,
        )

        preds = trainer.predict(model, dataloader)
        fns, names, preds = map(list, map(chain.from_iterable, zip(*preds)))
        preds = torch.stack(preds).cpu()  # .tolist()  # unbatch

        def mat2tree(pred, names):
            """Convert a matrix to a tree using BioNJ"""
            tree_NJ = sq.njtree(pred, names, algorithm=args.tree_alg).as_string(
                "newick"
            )
            pred = sq.squareform(pred)
            tree_BIONJ = BioNJ().reconstruct_tree(pred, names)
            return tree_NJ, tree_BIONJ

        with Parallel(args.workers, require="sharedmem") as parallel:
            pred_trees = parallel(delayed(mat2tree)(p, n) for p, n in zip(preds, names))
        pred_nj, pred_bionj = zip(*pred_trees)
        results = {
            "filenames": [fn.stem for fn in fns],
            "names": names,
            "predictions": preds,
            "trees_nj": pred_nj,
            "trees_bionj": pred_bionj,
        }
        torch.save(results, args.predict / args.out.with_suffix(".pt"))
        outfile = args.predict / args.out.with_suffix(".pd.gz")
        results["predictions"] = results["predictions"].tolist()
        pd.DataFrame(results).to_pickle(outfile)
        print(f"Predictions saved to {outfile}")

    else:
        print(model.policy_function)
        if not args.batch_size:  #
            # TODO: figure out where batch_size is saved in the checkpoint file --- and pretrained_filename is not None:
            init_bs = 512
            model.hparams.batch_size = init_bs
            tuner = Tuner(trainer)
            # o.w. tries to check val set and errors out with batch_size == 0
            optimal_batch_size = tuner.scale_batch_size(
                model, mode="power", init_val=init_bs, steps_per_trial=2
            )
            # optimal_learning_rate = tuner.lr_find(model)

            model.hparams.batch_size = optimal_batch_size
            # model.hparams.learning_rate = optimal_learning_rate

            print(optimal_batch_size)
        try:
            trainer.fit(model, ckpt_path=pretrained_filename)
        except KeyError as e:  # optimizer state dict not found
            print(e)
            trainer.fit(model)
        print("finished training model\n-----")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Train and test.  
                                     seqdir (training) or predict (testing) should contain individual alignment files.
                                     Alignments must be in phylip format and have the same number of taxa for batch_size > 1"""
    )
    parser.add_argument("--epochs", type=int, default=50, help="""train for n epochs""")
    parser.add_argument(
        "--polyak",
        type=float,
        default=None,
        help="""polyak averaging for target networks. If None, do not use polyak averaging.""",
    )
    parser.add_argument(
        "--check_interval",
        type=float,
        default=0.5,
        help="""check val loss every <frac> epoch.""",
    )
    parser.add_argument("--gpu", type=str, default=None, help="""gpu to use""")
    parser.add_argument(
        "--scale_y",
        type=int,
        default=None,
        help="""scale by 1/<scale_y>.  
                        if <scale_y>==1, scale trees to have unit height. """,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""batch size for training.  If None, Lightning will tune batch size""",
    )
    parser.add_argument(
        "--val_batches",
        type=int,
        default=None,
        help="""number of val batches.  
        if not set explicitly will be set to 300//train_batch_size""",
    )
    parser.add_argument(
        "--ngenes",
        type=int,
        default=1000,
        help="""number of genes per dataset (species tree for simphy)""",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="""stop if no improvement in val loss""",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="""print model summary""",
    )
    parser.add_argument("--debug", action="store_true", help="""debug mode""")
    parser.add_argument("--compile", action="store_true", help="""compile""")
    parser.add_argument(
        "--predict",
        type=Path,
        default=None,
        help="""predict ONLY sequences in this path""",
    )
    parser.add_argument(
        "--tree_alg",
        type=str,
        default="nj",
        choices=["nj", "bionj"],
        help="""algorithm to use for val/predict treebuilding.""",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("preds"),
        help="""output stem for output files (+.pd.gz)""",
    )
    parser.add_argument(
        "--phyloformer",
        action="store_true",
        help="""use pretrained phyloformer model""",
    )
    parser.add_argument(
        "--progressbar",
        action="store_true",
        help="""enable progress bar.
                        If False, will not show progress bar, but will still log training progress""",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="""by default, disable pin_memory.  
                        Otherwise OOM when we try to run a validation (and sometimes train) step.""",
    )
    parser.add_argument(
        "--workers",
        "-p",
        type=int,
        default=8,
        help="""num workers to prefetch data
                        (need >= 1.9gb each)""",
    )
    parser.add_argument(
        "--rootdir",
        type=Path,
        default=None,
        help="""dir containing training data""",
    )

    parser.add_argument(
        "--seqdir",
        type=Path,
        help="dir containing training seqs",
        default="seqs",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="""use automatic mixed precision. 
        Require Ampere based GPUs, such as A100s or 3090s. 
         slow on V100 (and occasionally leads to OOM), use only on A100 (small ~ 13% speedup)""",
    )
    parser.add_argument(
        "--valdir",
        type=Path,
        help="""directory path containing validation data""",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("/tmp"),
        help="""path to checkpoints""",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="""path to model""",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="""path to json-formatted config file.  Example:
        {
            "network": "equivariant",
            "regularizer":"von_neumann",
            "params": {
                "n_layers": 4,
                "n_heads": 4,
                "hidden_channels": 32,
                "axial": false,
            },
            "format": "distance",
            "loss": "mse",
            "learning_rate": 0.0005
        }
""",
        default=None,
    )

    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    main(args)
