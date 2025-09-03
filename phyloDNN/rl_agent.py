import imp
from tkinter import N
from numpy import cov
from torch import distributions
from typing import Union
import re
import subprocess
from pathlib import Path
import tempfile
from typing import List
try:
    from utils import covariance_to_distance, njtree
except ModuleNotFoundError:
    from .utils import covariance_to_distance, njtree

import torch
from joblib import Parallel, delayed
# from multiprocess import pool

nll_rx = re.compile('lh=-([\d.]+) ')


seqtype = 'DNA'

NPROCS = 8


def calculate_log_likelihood(seqfile: Path,
                             newicktrees: Union[str, list[str]],
                             iqtree: str = 'iqtree',
                             cores: int = 8) -> torch.Tensor:
    """returns nll of a set of tree topologies w.r.t. the given sequence file,
            optimizing the branch lengths."""
    if isinstance(newicktrees, str):
        newicktrees = [newicktrees]
    with tempfile.TemporaryDirectory() as outdir:
        outdir = Path(outdir)
        treefile = outdir/'tree.nw'
        outfile = outdir/'out'

        with open(treefile, 'w') as f:
            f.write('\n'.join(newicktrees))

        cmd = [iqtree,
               '-s', seqfile, '-z', treefile,
               '-n', '0', '-te', treefile, '--prefix', outfile,
               '-st', seqtype, '-nt', str(cores),
               '-fast',  # Turn on the fast tree search mode, where IQ-TREE will just
               # construct two starting trees: maximum parsimony and BIONJ, which
               # are then optimized by nearest neighbor interchange (NNI).
               # Introduced in version 1.6
               '-quiet']

        process = subprocess.run(cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        print(process)
        with open(outfile.with_suffix('.trees')) as tree_scores:
            nll = list(map(float, nll_rx.findall(tree_scores.read())))
    return torch.Tensor(nll)


def step(seqfile: Path,
         n_samples=10):
    """take a policy gradient step.

    Args:
        seqfile (Path): path to torch data file
        n_samples (int, optional): number of samples for each sequence. Defaults to 10.
    """

    embedding_size = 100
    alignment = load_alignment(seqfile)
    V = model(alignment) / embedding_size
    # expectation of m = df*covariance_matrix, variance = df*(V_ij^2+V_ii*V_jj)
    m = distributions.wishart.Wishart(df=embedding_size, covariance_matrix=V)
    covariances = m.sample(sample_shape=torch.Size([n_samples]))
    distances = Parallel(n_jobs=NPROCS)(
        delayed(njtree)(covariance_to_distance(c)) for c in covariances)
    rewards = calculate_log_likelihood(seqfile, distances)
    baseline = rewards.mean()
    loss = -torch.mean(m.log_prob(covariances) * (rewards-baseline))
    loss.backward()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test.')
    parser.add_argument('--save_freq',
                        type=int,
                        default=200,
                        help='''save every n batches/epochs''')
    parser.add_argument('--save_every',
                        nargs='?',
                        choices=['epoch', 'batch'],
                        default='epoch',
                        help='''count batches or epochs - UNUSED''')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='''train for n epochs''')
    parser.add_argument('--test_tree_size',
                        type=int,
                        default=10,
                        help='''(fixed) size of test trees''')
    parser.add_argument('--max_taxa',
                        type=int,
                        default=6,
                        help='''max height of alignment (inclusive)''')
    parser.add_argument('--ngenes',
                        type=int,
                        default=1000,
                        help='''number of genes per dataset (species tree for simphy)''')
    parser.add_argument('--max_length',
                        type=float,
                        default=3e3,
                        help='''max length of alignment (inclusive)''')
    parser.add_argument('--min_taxa',
                        type=int,
                        default=4,
                        help='''min width of alignment (inclusive)''')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='''batch size''')
    parser.add_argument('--workers',
                        '-p',
                        type=int,
                        default=2,
                        help='''num workers to prefetch data
                        (need >= 1.9gb each)''')
    parser.add_argument('--loss',
                        '-l',
                        type=str,
                        default='huber',
                        choices=['mse', 'huber'],
                        help='''loss function''')
    parser.add_argument('--regularizer',
                        '-r',
                        type=str,
                        default='logdet',
                        choices=['logdet', 'von_neumann'],
                        help='''PSD Bregman divergence regularizer''')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='extra verbosity for debugging')
    parser.add_argument('--matrix_type',
                        type=str,
                        default='covariance',)
    parser.add_argument('--reset_optimizer',
                        action='store_true',
                        help='load model from checkpoint file but reset Adam')
    parser.add_argument('--overwrite',
                        action='store_true',
                        help='overwrite data files')
    parser.add_argument('--seqdir',
                        type=str,
                        help='directory containing raw, processed dirs',
                        default='/N/project/phyloML/simphy/sim20_genes')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='''path to checkpoints;
                        will append ckpt to path''',
                        default='/N/project/phyloML/trained_models/gat_2layer.')

    args = parser.parse_args()

    modelfile, optfile, checkpointfile, configfile = (
        args.checkpoint+x for x in ('model', 'optimizer', 'ckpt', 'config'))

    if args.config:
        configfile = args.config
    if configfile.exists():
        print(f'loading params from {configfile}')
        from json import load
        with open(configfile) as f:
            config = load(f)
        model_params = config['model']
        optimizer_params = config['optimizer']
    print('model:', model_params, '\n---\noptimizer:', optimizer_params)
    model = build_model(multigpu=args.multigpu,
                        device=device, **model_params)
    opt = build_optimizer(model, **optimizer_params)

    if not args.train_from_scratch and os.path.exists(checkpointfile):
        model, opt = load_train_state(checkpointfile, model, opt, device)
        if args.reset_optimizer:
            opt = build_optimizer(model, **optimizer_params)
        if args.verbose:
            print('loaded weights...')
