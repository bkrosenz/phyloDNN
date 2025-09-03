
import numba
import numpy as np
import torch
import skbio
from .seq_utils import njtree

__package__ = 'phyloDNN'


import warnings

warnings.filterwarnings('ignore')


def cov2tree(cov, ids):
    c = covariance_to_distance(cov)
    d = skbio.DistanceMatrix(c, map(str, ids))
    tree = skbio.tree.nj(d, result_constructor=str)
    return tree


@numba.njit(parallel=True)
def covariance_to_distance(C):
    """convert a phylogenetic covariance matrix to a tree distance matrix """
    D = np.empty_like(C)
    n, m = C.shape
    for i in range(n):
        for j in range(m):
            D[i, j] = C[i, i]+C[j, j]-2*C[i, j]
    return D


def make_tree_from_submatrix(y: torch.Tensor, idx, taxon_namespace=None):
    ypred_batch = y[idx, :][:, idx]
    if ypred_batch.diag().sum() > 0:
        ypred_batch = covariance_to_distance(ypred_batch.numpy())
    if (ypred_batch < 0).any():
        raise ValueError(f'negative distance found in distance matrix')

    # must convert to tuple so that lru_cache hash is actually meaningful

    pred_tree = njtree(tuple(map(tuple, ypred_batch)),
                       taxon_namespace)
    return pred_tree
