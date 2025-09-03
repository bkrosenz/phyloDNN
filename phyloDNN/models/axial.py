import argparse
import math
import re
import subprocess
import tempfile
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Iterable, List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from joblib import Parallel, delayed
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import (BatchSizeFinder, LearningRateMonitor,
                                         ModelCheckpoint)
from torch import nn
from torch.cuda.amp import autocast

try:

    from ..seq_utils import (alignment_to_torch, make_tree, njtree,
                             rf_distance, wrf_distance)
    from .graph_utils import (EMBED_DIM, CovarianceDecoder, EmbedLayer,
                              MetricDecoder, build_fc_network, make_conv_net)

except:
    from graph_utils import (EMBED_DIM, EmbedLayer, MetricDecoder,
                             build_fc_network, make_conv_net)
    from seq_utils import (alignment_to_torch, make_tree, njtree, rf_distance,
                           wrf_distance)

iqtree_rx = re.compile('lh=(-[\d.]+) ')
raxml_rx = re.compile('Final LogLikelihood: (-[\d.]+)')
raxml_rx_multiple = re.compile('final logLikelihood: (-[\d.]+)')


seqtype = 'AA'  # 'DNA'


def cov2tree(cov, ids):
    import skbio

    from phyloDNN.covariance import covariance_to_distance
    c = covariance_to_distance(cov)
    d = skbio.DistanceMatrix(c, map(str, ids))
    return skbio.tree.nj(d, result_constructor=str)


def calculate_log_likelihood(seqfile: Path,
                             newicktrees: Union[str, list[str]] = None,
                             covariances: torch.Tensor = None,
                             ids: Iterable = None,
                             engine: str = 'raxml',
                             seqtype: Literal['AA', 'DNA'] = 'AA',
                             verbose: bool = False,
                             optimize: bool = True,
                             cores: int = 8) -> torch.Tensor:
    """returns ll of a set of tree topologies w.r.t. the given sequence file,
            optimizing the branch lengths.
            The returned LL ignores the scale of the input trees."""
    if newicktrees is None and covariances is not None:
        newicktrees = Parallel(n_jobs=cores)(
            delayed(cov2tree)(c.cpu().numpy(), ids) for c in covariances)

    if isinstance(newicktrees, str):
        newicktrees = [newicktrees]
    if verbose:
        print(newicktrees, seqfile)
    with tempfile.TemporaryDirectory() as outdir:
        outdir = Path(outdir)
        treefile = outdir/'tree.nw'
        outfile = outdir/'out'
        # print('\n'.join(newicktrees[:10]))
        with open(treefile, 'w') as f:
            f.write('\n'.join(newicktrees))
        exit_code = 1
        # Loop until we get the right number of cores
        while exit_code != 0 and cores > 0:
            if engine == 'iqtree':

                cmd = ['iqtree',
                       '-s', seqfile, '-te', treefile, '-z',
                       treefile, '--prefix', outfile,
                       '-st', seqtype, '-nt', cores,
                       # set -n 0 to avoid tree search and just perform tree topology tests.
                       '-n', '0',
                       '--fast',  # Turn on the fast tree search mode, where IQ-TREE will just
                       # construct two starting trees: maximum parsimony and BIONJ, which
                       # are then optimized by nearest neighbor interchange (NNI).
                       # Introduced in version 1.6
                       '--safe',  # Safe likelihood kernel to avoid numerical underflow
                       '--quiet']
            else:  # assume raxml-ng
                # replace --loglh with --evaluate to optimize branch lengths and model params
                cmd = ['raxml-ng', '--msa', seqfile,
                       '--data-type', 'AA',
                       '--prefix', outfile, '--model', 'LG4M',
                       '--tree', treefile,
                       '--threads', cores]
                if optimize:
                    cmd.append('--evaluate')
                else:
                    cmd.append('--loglh')
            cmd = map(str, cmd)
            process = subprocess.run(cmd,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            exit_code = process.returncode
            if exit_code != 0:
                if b"You have specified more threads than CPU cores available" in process.stderr or \
                    b"There are fewer alignment sites" in process.stderr or \
                        b"CPU core oversubscription detected" in process.stdout:
                    cores //= 2
                else:
                    print(process.stdout, process.stderr)
                    raise (subprocess.CalledProcessError(
                        exit_code, ' '.join(cmd)))
        # print(process, list(outdir.iterdir()), ' '.join(map(str, cmd)))
        if engine == 'iqtree':
            rx = iqtree_rx
            with open(outfile.with_suffix('.trees'), 'r') as tree_scores:
                scores = tree_scores.read()

        else:
            rx = raxml_rx if len(newicktrees) == 1 else raxml_rx_multiple
            scores = process.stdout.decode()
    return torch.Tensor(list(map(float, rx.findall(scores))))


def normalize_loss(loss: Callable):
    def new_loss(pred: torch.Tensor, true: torch.Tensor):
        '''must use sum reduction; with MSE, an n-batch of 1 n-graph has lower loss than an n-batch of 2 n/2-graphs.
        Therefore, we normalize by the number of actual edges in all the graphs in the batch (multiply by 2 because of symmetry).'''
        with autocast(enabled=False):
            return loss(pred.double(), true.double())/true.count_nonzero()*2
    return new_loss


def batch_iter(mat, batches, square=False):
    """iterates through submatrices of mat indexed by values in batches"""
    for b in torch.unique(batches):
        ix = batches == b
        if square:
            yield mat[ix][:, ix]
        else:
            yield mat[ix]


class FineTuneBatchSizeFinder(BatchSizeFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.scale_batch_size(trainer, pl_module)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

        self.parameter = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]*self.parameter
        return x


class Attention(nn.Module):
    def __init__(self, max_length,
                 embed_dim=6,
                 hidden_dim=256,
                 kernel_size=32,
                 stride=5,
                 dropout=0):
        from phyloDNN.models import KernelAxialMultiAttention
        super().__init__()
        self.add_module('embed',
                        EmbedLayer(embed_dim, dropout))
        self.add_module('conv', nn.Conv1d(
            embed_dim, hidden_dim, kernel_size, stride=stride))
        self.add_module('pos_embed', PositionalEncoding(hidden_dim))
        # self.add_module('pos_embed',
        #                 AxialPositionalEmbedding(
        #                     dim=hidden_dim,
        #                     shape=(max_length, ),
        #                     emb_dim_index=-1
        #                 ))
        self.add_module('attn',
                        KernelAxialMultiAttention(
                            h_dim=hidden_dim,
                            n_heads=8,
                            # num_odimensions=2,
                            # dim_index=-1
                        ))

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        with autocast():
            x = data.x.to(self.device)
            batches = data.batch.to(self.device)
            x = self.embed(x).squeeze()
            x = self.conv(x).transpose(-2, -1)
            x = self.pos_embed(x)
            output = []
            for batch in batch_iter(x, batches):
                output.append(self.attn(batch[None, ...]).squeeze())
            output = torch.cat(output, 0)
        return output


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    @lru_cache(maxsize=16)
    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TransformerPredictor(LightningModule):
    def __init__(self,
                 lr, warmup,
                 max_iters,
                 embed_dim=6,
                 hidden_dim=256,
                 conv_layer_sizes=[256, 256],
                 fc_layers=[1024, 512, 256],
                 kernel_size=32,
                 n_attention_layers=1,
                 stride=5,
                 dropout=0,
                 input_dropout=0.0,
                 decoder='metric',
                 loss='mse', **kwargs):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        from phyloDNN.models import KernelAxialMultiAttention
        super().__init__()
        self.add_module(
            'embed',
            EmbedLayer(self.hparams.embed_dim, self.hparams.dropout))
        self.add_module(
            'conv',
            make_conv_net(
                conv_layer_sizes=[self.hparams.embed_dim] +
                self.hparams.conv_layer_sizes+[self.hparams.hidden_dim],
                kernel=self.hparams.kernel_size,
                stride=self.hparams.stride,
                batch_norm=True,
                dropout=self.hparams.dropout),
        )
        self.add_module(
            'pos_embed',
            PositionalEncoding(
                self.hparams.hidden_dim))
        attention_layers = [KernelAxialMultiAttention(
            dim=self.hparams.hidden_dim,
            heads=8,
            num_dimensions=2,
            dim_index=-1
        ) for _ in range(self.hparams.n_attention_layers)]
        attention_layers = nn.Sequential(*attention_layers)
        self.add_module(
            'attn', attention_layers)

        fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.AdaptiveAvgPool1d(
                self.hparams.fc_layers[0]),
            *build_fc_network(
                layers=self.hparams.fc_layers,
                nonlinearity=nn.ELU,
                batch_norm=False),
        )
        self.add_module('fc_layer', fc_layer)
        if self.hparams.decoder == 'metric':
            self.add_module('decoder', MetricDecoder())
        elif self.hparams.decoder == 'covariance':
            self.add_module('decoder', CovarianceDecoder())
        # else:
        #     self.add_module('decoder', decoder)

        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if self.hparams.loss == 'mse':
            self.criterion = normalize_loss(torch.nn.MSELoss(reduction='sum'))
        elif self.hparams.loss == 'smooth_l1':
            self.criterion = normalize_loss(
                torch.nn.SmoothL1Loss(
                    beta=2, reduction='sum')
            )

    def forward(self, data=None, x=None, batches=None, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        with autocast():
            if data is not None:
                batches = data.batch.to(self.device, non_blocking=True)
                x = data.x.to(self.device)

            x = self.embed(x).squeeze()
            x = self.conv(x).transpose(-2, -1)
            x = self.pos_embed(x)
            output = []
            n = 0
            for batch in batch_iter(x, batches):
                output.append(self.attn(batch[None, ...]).squeeze())
                n += 1
            output = torch.cat(output, 0)
            output = self.fc_layer(output)
        output = self.decoder(output.float(), batches)
        return output

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors

        y = batch.y.to(self.device)
        # Perform prediction and calculate loss and accuracy
        preds = self.forward(data=batch)
        loss = self.criterion(preds, y)
        n = len(batch.batch.unique())
        # Logging
        self.log(f"{mode}_loss", loss, batch_size=n)
        # self.log(f"{mode}_acc", acc)
        return loss

    def _acc(self, data, mode='val', weighted=False):
        from phyloDNN import seq_utils as su
        ypred = self.forward(data=data).cpu()
        b = data.batch.cpu()
        y = data.y.cpu()
        calculate_dRF = su.wrf_distance if weighted else partial(
            su.rf_distance, normalize=True)
        n = d = 0.
        for l in b.unique():
            idx = b == l
            ypred_batch = ypred[idx, :][:, idx]
            ytrue_batch = y[idx, :][:, idx]
            true_tree = su.njtree(tuple(map(tuple, ytrue_batch)))
            pred_tree = su.njtree(tuple(map(tuple, ypred_batch)),
                                  true_tree.taxon_namespace)
            d += calculate_dRF(pred_tree, true_tree)
            n += 1
        d /= n
        self.log(f"{mode}_drf", d, batch_size=n)

        return d

    def training_step(self, batch):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, *args):
        _ = self._acc(batch, mode="val")

    def test_step(self, batch, *args):
        _ = self._acc(batch, mode="test")


class QFunction(TransformerPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.hparams.fc_layers[-1], 1))
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def forward_list(self, batch):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
        """
        if not isinstance(batch, list):
            batch = [batch]
        res = torch.empty(len(batch), device=batch.device)
        for i, x in enumerate(batch):
            with autocast():
                x = self.embed(x).squeeze()
                x = self.conv(x).transpose(-2, -1)
                x = self.pos_embed(x)
                x = self.attn(x[None, ...]).squeeze()
                x = self.fc_layer(x)
                x = x.sum(0)
                x = self.decoder(x)
            res[i] = x.float()
        return res.squeeze()

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors

        y = calculate_log_likelihood(
            seqfile, list(batch_iter(batch.y, batch.batch))).to(self.device, non_blocking=True)

        # Perform prediction and calculate loss and accuracy
        preds = self.forward(data=batch)
        loss = self.criterion(preds, y)
        n = len(batch.batch.unique())
        # Logging
        self.log(f"{mode}_loss", loss, batch_size=n)
        # self.log(f"{mode}_acc", acc)
        return loss


class ValueFunction(TransformerPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.hparams.fc_layers[-1], 1))
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def forward(self, x):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
        """
        with autocast():
            x = self.embed(x).squeeze()
            x = self.conv(x).transpose(-2, -1)
            x = self.pos_embed(x)
            x = self.attn(x[None, ...]).squeeze()
            x = self.fc_layer(x)
            x = x.sum(0)
            x = self.decoder(x)
        return x.float()

    def _calculate_loss(self, batch, mode="train"):
        # Fetch data and transform categories to one-hot vectors

        y = calculate_log_likelihood(
            seqfile, list(batch_iter(batch.y, batch.batch))).to(self.device, non_blocking=True)

        # Perform prediction and calculate loss and accuracy
        preds = self.forward(data=batch)
        loss = self.criterion(preds, y)
        n = len(batch.batch.unique())
        # Logging
        self.log(f"{mode}_loss", loss, batch_size=n)
        # self.log(f"{mode}_acc", acc)
        return loss

    # def _calculate_loss(self,)
