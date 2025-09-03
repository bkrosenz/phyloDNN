import math
import os
import subprocess as sp
from collections import defaultdict
from functools import lru_cache, partial, reduce
from itertools import islice, zip_longest
from time import time
from typing import Union

import numpy as np
import torch
from pandas import DataFrame
from torch import (Tensor, _mkldnn, load, no_grad, save,
                   sparse_coo_tensor, triu_indices, unique)
from torch.nn import Module

import phyloDNN.seq_utils as sq

__package__ = 'phyloDNN'


def squareform(arr: torch.Tensor, n=None, add_diagonal=False):
    """convert list of upper diagonal entries to symmetric matrix, optional initial batch dimension.
    Idempotent.

    Args:
        arr (_type_): input array of shape npairs or nbatches x npairs
        n (_type_, optional): size. if not specified will be inferred. Defaults to None.
        add_diagonal (bool, optional): If True, will add zeros on the diagonal.  Use ONLY for distance matrices.  
        If False, assumes that diagonal is included. Defaults to False.

    Returns:
        _type_: if batch
    """

    if arr.dim() == 2:
        b, l = arr.shape
        if b == l and torch.all(arr == arr.t()):
            return arr  # already square
    elif arr.dim() == 3:
        b, r, c = arr.shape
        if r == c and torch.all(arr == arr.transpose(-1, -2)):
            return arr  # already square
    else:
        l = len(arr)
        b = 0

    if n is None:
        n = int((np.sqrt(8*l+1)+1)/2) - 1+add_diagonal  # quadratic formula

    ix = torch.triu_indices(n, n, offset=int(add_diagonal))
    if b:
        bix = torch.arange(b, dtype=ix.dtype).repeat_interleave(
            ix.shape[-1]).unsqueeze(0)
        ix = torch.concat((bix, ix.repeat((1, b))))
        x = torch.sparse_coo_tensor(
            ix.to(arr.device),
            arr.ravel(),
            size=(b, n, n)).to_dense()
    else:
        x = torch.sparse_coo_tensor(ix.to(arr.device),
                                    arr, size=(n, n)).to_dense()
    return x+x.transpose(-1, -2)


def trainable_parameters(model):
    return sum(p.numel()
               for p in model.parameters() if p.requires_grad)


def validation_step(model: Module,
                    data: Tensor,
                    dtype: str = 'distance',
                    return_trees: bool = False,
                    hamming_nj: bool = False,
                    print_all: bool = True):
    '''expensive step: builds a njtree for the pseudo-distance matrix x_hat'''
    from dendropy.calculate import treecompare
    from torch import cuda
    model.eval()
    with no_grad():
        ypred = model(data).cpu()
    b = data.batch.cpu()
    y = data.y.cpu()
    cuda.empty_cache()
    stats = defaultdict(lambda: 0)
    n = 0.
    trees = []
    hamming_trees = []
    for l in b.unique():
        idx = b == l
        ypred_batch = ypred[idx, :][:, idx]
        ytrue_batch = y[idx, :][:, idx]
        try:
            if dtype == 'covariance':
                ytrue_batch = sq.covariance_to_distance(ytrue_batch.numpy())
                ypred_batch = sq.covariance_to_distance(ypred_batch.numpy())
            if (ytrue_batch < 0).any() or (ypred_batch < 0).any():
                raise ValueError(f'negative distance found in distance matrix')

            # must convert to tuple so that lru_cache hash is actually meaningful
            true_tree = sq.njtree(tuple(map(tuple, ytrue_batch)))
            pred_tree = sq.njtree(tuple(map(tuple, ypred_batch)),
                               true_tree.taxon_namespace)

        except:
            # continue
            raise
        stats['uw_dist'] += rf_distance(pred_tree, true_tree)
        stats['w_dist'] += wrf_distance(pred_tree, true_tree)
        if hamming_nj:
            h_tree = njtree(
                torch.pdist(data.x[idx].squeeze().float(),
                            p=1),
                true_tree.taxon_namespace)
            stats['h_uw_dist'] = rf_distance(h_tree, true_tree)
            stats['h_w_dist'] = wrf_distance(h_tree, true_tree)
        ynorm = np.linalg.norm(ypred_batch)

        rf_p, rf_tail = yh_prob(true_tree, stats['uw_dist'])
        # if print_all:
        #     print(         f'{stats}}')
        stats['rf_tails'] -= np.log(rf_tail)
        stats['ynorm'] += ynorm
        stats['zero_preds'] += not ynorm
        n += 1
        if return_trees:
            trees.append((str(true_tree), str(pred_tree)))
    for k in stats:
        if k not in {'rf_tails', 'zero_preds'}:
            stats[k] /= n
    if return_trees:
        stats['trees'] = DataFrame(trees, columns=['true_tree', 'pred_tree'])

    return stats


def validation_step_list(model,
                         data,
                         return_trees: bool = False,
                         ):
    '''expensive step: builds a njtree for the pseudo-distance matrix x_hat'''
    from dendropy.calculate import treecompare
    rf_tails = uw_dist_total = w_dist_total = ynorm_total = n = zero_preds = 0.
    for y, ypred in zip(data.y, model(data)):
        y = y.cpu()
        ypred = ypred.cpu()
        try:
            true_tree = njtree(y)
            pred_tree = njtree(ypred, true_tree.taxon_namespace)
        except:
            continue
        uw_dist = rf_distance(pred_tree, true_tree, normalize=True)
        w_dist = wrf_distance(pred_tree, true_tree)
        ynorm = ypred.norm()

        rf_p, rf_tail = yh_prob(true_tree,  rf_distance(
            pred_tree, true_tree, normalize=False))
        print(
            f'{len(true_tree)}\t{uw_dist}\t{rf_tail:.4f}\t{w_dist:.3f}\t\t{ynorm:.4f}')
        rf_tails -= np.log(rf_tail) if rf_tail != 0 else 0
        w_dist_total += w_dist
        uw_dist_total += uw_dist
        ynorm_total += ynorm
        zero_preds += not ynorm
        n += 1
    return rf_tails, uw_dist_total/n, w_dist_total/n, ynorm_total/n, zero_preds


@lru_cache(maxsize=20)
def double_factorial(n):
    return math.prod(range(n, 0, -2))


def summarize(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              dtype: str,
              now: float = -1,
              as_list: bool = True,
              n_batches: int = 5,
              return_trees: bool = False,
              quiet: bool = False,
              print_all=False) -> Union[DataFrame, None]:
    '''Summarize performance metrics. This normalizes 
    by number of instances, but not by the size of the trees.

    Args:
        model (torch.nn.Module):    GNN 
        data_loader (torch.utils.data.DataLoader): _description_
        dtype (str): distance or covariance 
        now (float, optional): _description_. Defaults to -1.
        as_list (bool, optional): _description_. Defaults to True.
        n_batches (int, optional): If None, summarize entire dataset. Defaults to 5.
        return_trees (bool, optional): _description_. Defaults to False.
        quiet (bool, optional): _description_. Defaults to False.
        print_all (bool, optional): _description_. Defaults to False.

    Returns:
        Union[DataFrame, None]: _description_
    '''
    import pandas as pd
    if as_list:
        val_step = validation_step_list
    else:
        val_step = partial(validation_step,
                           dtype=dtype,
                           return_trees=return_trees,
                           hamming_nj=True,
                           print_all=print_all)
    try:
        if print_all:
            print(f'ntaxa\td_RF\td_wRF\tPr(D <= d_RF)\t\tynorm\n------')
        if n_batches is None:
            vals = [val_step(model, data) for data in data_loader]
            n_batches = 1
        else:
            vals = [val_step(model, data)
                    for data in islice(data_loader, n_batches)]
        vals = pd.DataFrame(vals)
        if return_trees:
            trees = vals['trees']
            vals.drop(columns='trees', inplace=True)
        if not quiet:
            summary_stats = vals.sum()
            n_trees = double_factorial(2*len(next(iter(data_loader)).y)-5)
            print(
                f'''neg log Pr: {summary_stats['rf_tails']/n_batches:.4f}\t
                    avg d_RF (Hamming): {summary_stats['h_uw_dist']/n_batches:.4f}\t
                    avg d_wRF (Hamming): {summary_stats['h_w_dist']/n_batches:.4f}\t
                    avg d_RF: {summary_stats['uw_dist']/n_batches:.4f} (Expected: {1-1./n_trees}\t
                    avg d_wRF: {summary_stats['w_dist']/n_batches:.4f}\t
                    \tavg pred norm: {summary_stats['ynorm']/n_batches:.4f}\t
                    zeros: {summary_stats['zero_preds']}''')
            if now > -1:
                batch_time = (time()-now)
                print(
                    f'''time: {batch_time:.4f} s.'''
                )
    except Exception as e:
        raise (e)
    if return_trees:
        return pd.concat(trees.to_list())


def get_gpu_memory():
    def _output_to_list(x): return x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0])
                          for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values


def compose(*funcs):
    'compose functions fn(...f2(f1(x))...)'
    return lambda x: reduce(lambda f, g: g(f), funcs, x)


def array_to_mat(ix, arr, **kwargs):
    return sparse_coo_tensor(ix, arr, **kwargs).to_dense()


def upper_triangular(array) -> Tensor:
    """Convert square form to long form

    Args:
        array (Tensor): symmetric distance matrix

    Returns:
        Tensor: n x (n-1) length array
    """
    n = array.shape[0]
    ix = triu_indices(n, n, offset=1)
    if isinstance(array, Tensor):
        ix = ix.to(array.device)
    return array[ix[0], ix[1]]


# def squareform(arr, n=None):
#     if n is None:
#         n = int((np.sqrt(8*len(arr)+1)+1)/2)  # quadratic formula
#     ix = triu_indices(n, n, offset=1)
#     if isinstance(arr, Tensor):
#         ix = ix.to(arr.device)
#     x = sparse_coo_tensor(ix, arr, size=(n, n)).to_dense()
#     return x+x.T


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return list(zip_longest(*args, fillvalue=fillvalue))


def graph_to_matrix(data):
    '''convert edge,value format to dense adjacency matrix'''
    return array_to_mat(data.edge_index, data.y.squeeze())


# @lru_cache


def load_optimizer_state(file_path: os.PathLike,
                         opt: torch.optim.Optimizer,
                         device=None,
                         ) -> torch.optim.Optimizer:
    """Load an optimizer
    Args:
        file_path (os.PathLike): path
        opt (torch.optim.Optimizer): optimizer object with same parameters as saved state
        device (_type_, optional): Put ALL params on this device. Defaults to None.
    Returns:
        torch.optim.Optimizer: _description_
    """
    checkpoint = load(file_path, map_location=device)  # torch.device('cpu'))
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    return opt


def load_model_state(file_path: os.PathLike,
                     model: torch.nn.Module,
                     device=None,
                     ) -> torch.nn.Module:
    """load model

    Args:
        file_path (os.PathLike): path
        model (torch.nn.Module): DNN
        device (_type_, optional): if not None, will put all parameters on device.
        Calling set_parameters will move embed to cpu, but then optimizer must be reinitialized. Defaults to None.
        NOTE: to change device of model, must create via the constructor, not loading a torch.save'd file
    Returns:
        torch.nn.Module: _description_
    """
    # , map_location=device) # need map location if saved from different device
    checkpoint = load(file_path, map_location=device)  # torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def load_train_state(file_path,
                     model,
                     opt,
                     device,
                     opt_cpu=None):
    '''NOTE: to change device of model, must create via the constructor, not loading a torch.save'd file'''
    # , map_location=device) # need map location if saved from different device
    checkpoint = load(file_path, map_location=device)  # torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    if opt_cpu is not None:
        opt_cpu = opt_cpu.load_state_dict(
            checkpoint['optimizer_cpu_state_dict'])
        return model, opt, opt_cpu

    return model,  opt


def save_train_state(name, mod, opt, opt_cpu=None):
    state = mod.state_dict()
    for k, v in state.items():
        if v.layout == _mkldnn:
            state[k] = v.to_dense()
    params = {
        'model_state_dict': state,
        'optimizer_state_dict': opt.state_dict(),
    }
    if opt_cpu is not None:
        params['optimizer_cpu_state_dict'] = opt_cpu.state_dict()

    save(
        params,
        name
    )


def batch_iter(mat, batches, square=False):
    """iterates through submatrices of mat indexed by values in batches"""
    for b in unique(batches):
        ix = batches == b
        if square:
            yield mat[ix][:, ix]
        else:
            yield mat[ix]


def make_network(network,preprocessor=None,**kwargs):
    try:
        from graph_nets import (GAT, AttentionNet,AttentionGAT, AxialGAT, AxialGCN,
                                  FreqNet, FreqNet2, FreqPairNet, LocalNet,
                                  SDPANet, SeqGAT, SeqNet, SeqPairNet)
    except:
        from .graph_nets import (GAT, AttentionNet,AttentionGAT, AxialGAT, AxialGCN,
                                  FreqNet, FreqNet2, FreqPairNet, LocalNet,
                                  SDPANet, SeqGAT, SeqNet, SeqPairNet,
                                  EquivariantNet)
    if network == 'compressed_attention':
            net = CompressedAttentionNet(**kwargs)
    elif network == 'sdpa':
        net = SDPANet(
            **kwargs,
            preprocessor=preprocessor,
        )
    elif network == 'local':
        net = LocalNet(
            **kwargs,
            preprocessor=preprocessor,
        )
    elif network == 'equivariant':
        net = EquivariantNet(
            **kwargs,
            preprocessor=preprocessor,
        )
    elif network == 'seq':
        net = SeqNet(
            **kwargs,
            preprocessor=preprocessor,
        )
    elif network == 'dual':
        net = SeqPairNet(
            **kwargs,
            preprocessor=preprocessor,
        )
    elif network == 'attention':
        net = AttentionNet(
            **kwargs,
            preprocessor=preprocessor,
        )
        # with default params, AttentionNet should converge to vall acc .066 on 20-taxa aligns, and .096 on 60-taxa aligns after < 58000 samples
        # max batch size: 16
    elif network == 'freq-net':
        net = FreqNet(**kwargs)
    elif network == 'freq':
        net = FreqNet2(**kwargs)
    elif network == 'freq-pair':
        net = FreqPairNet(**kwargs)
    elif network == 'gat':
        net = GAT(**kwargs)
    elif network == 'axial-gat':
        net = AxialGAT(**kwargs)
        # converge to xx after xx
        # batch size: 256
    elif network == 'attn-gat':
        net = AttentionGAT(**kwargs)
    elif network == 'seq-gat':
        net = SeqGAT(**kwargs)
    elif network == 'double-block-gcn':
        net = DoubleBlockConvNetwork(**kwargs)
    elif network == 'axial-gcn':
        net = AxialGCN(**kwargs)
    else:
        net = build_model(model_type=network, **kwargs)
    return net


def build_model(
        device: str = 'cpu',
        multigpu: bool = False,
        model_type: str = 'gat',
        **model_params):
    if model_type == 'gat':
        from .models import AttentionNetwork
        model = AttentionNetwork
    elif model_type == 'gcn':
        from .models import EdgeConvNet
        model = EdgeConvNet
    elif model_type == 'gat-seq':
        from .models import SequentialAttentionNetwork
        model = SequentialAttentionNetwork
    elif model_type == 'double-gat':
        from .models import GATBlockConvNetwork
        model = GATBlockConvNetwork
    elif model_type == 'dynamic-gcn':
        from .models import DynamicConvNetwork
        model = DynamicConvNetwork
    elif model_type == 'block-gcn':
        from .models import BlockConvNetwork
        model = BlockConvNetwork
    elif model_type == 'double-block-gcn':
        from .models import DoubleBlockConvNetwork
        model = DoubleBlockConvNetwork
    elif model_type == 'double-gcn':
        from .models import DoubleDynamicConvNetwork
        model = DoubleDynamicConvNetwork
    elif model_type == 'gmt':
        from .models import SequentialGMTNetwork
        model = SequentialGMTNetwork
    elif model_type == 'res-gcn':
        from .models import ResidualEdgeConvNet
        model = ResidualEdgeConvNet
    else:
        raise ValueError(f'model type {model_type} not recognized')
    return model(
        **model_params,
        device=device)


def build_optimizer(model,
                    lr: float = 0.001,
                    weight_decay=.01):
    from torch.optim import AdamW
    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-5)
    return optimizer


def distortion_np(X, Y):
    '''given two distance matrices (arrays), calculate the distortion'''
    R = X/Y
    R = R[np.logical_and(~np.isnan(R), R > 0)]
    return R.max() / R.min()


def distortion(X, Y):
    '''given two distance matrices (tensors), calculate the distortion'''
    from torch import logical_and
    R = X/Y
    R = R[logical_and(~R.isnan(), R > 0)]
    return R.max() / R.min()
