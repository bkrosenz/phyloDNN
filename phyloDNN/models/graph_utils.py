from torch import nn
from typing import List, Union
from functools import lru_cache, partial
from itertools import chain, repeat

import torch
from torch.nn import Module
from scipy.spatial.distance import squareform

from ..utils import array_to_mat, batch_iter
from .resnet import (large_cnn, resnet18, resnet34, resnet50, resnet101,
                     simple_cnn)

resnet_models = dict(resnet18=resnet18, resnet34=resnet34,
                     resnet50=resnet50, resnet101=resnet101,
                     cnn=simple_cnn,
                     large_cnn=large_cnn)

# torch.distributed.init_process_group()


LETTERS = ('-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
           'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
# there are 16 degenerate IUPAC DNA codes and 20 AA + 1 for gaps + 1 for zero padding
N_STATES = len(LETTERS)+1

EMBED_DIM = 6


# @lru_cache
def make_batch_indices(n_seqs: int, batch_size: int,device=None):
    """make indices for n_seqs*batch_size sequences

    Args:
        n_seqs (int): seqs per
        batch_size (int): alignments per batch
    Returns:
        Tensor: _description_
    """
    return torch.arange(0, batch_size,device=device).repeat_interleave(n_seqs).long() #torch.range(batch_size).repeat_interleave()


@lru_cache
def make_indices(batch_size: int, n_seqs: int, directed=True,device=None):
    """Returns edge indices for one batch of N=<batch_size> separate fully connected K=<n_seqs>-graphs

    Args:
        batch_size (int): 1st dim
        n_seqs (int): will generate (n_seqs \choose 2) edges
        directed (bool, optional): If False, will double the edge list to include forward and backward edges. Defaults to True.

    Returns:
        _type_: _description_
    """
    indices = torch.triu_indices(n_seqs, n_seqs, 1,device=device)
    if not directed:
        indices = torch.hstack([indices, indices.flip(0)])
    batch_idx = (torch
                 .arange(0, batch_size,device=device)
                 .repeat_interleave(indices.shape[1])
                 .repeat((2, 1))
                 .long())*n_seqs
    return indices.repeat((1, batch_size))+batch_idx


def output_hook(self, grad_input, grad_output):
    self.grad_output = grad_output


class GraphNetwork(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_devices(self, device=None):
        if device is not None:
            self.device = device
        for name, layer in self.named_children():
            # name == 'gene_encoder' or name == 'set_encoder':
            if name == 'embed' or name == 'char_embed':
                layer.cpu()
            elif 'device1' in self.__dict__ and name in set(['gene_encoder', 'elu', 'norm1']):
                layer.to(self.device1)
            else:
                layer.to(device)

    def rf_distance(self,
                    input: torch.Tensor,
                    batches: torch.Tensor,
                    output,
                    ):
        from .. import utils as u
        d = n = 0
        if isinstance(output, torch.Tensor):  # comparing many to many
            yiter = map(u.njtree, u.batch_iter(output, batches=batches))
        elif isinstance(output, u.Tree):  # comparing many trees to one
            yiter = repeat(output)
        else:
            raise ValueError
        for x, true_tree in zip(u.batch_iter(input, batches=batches), yiter):
            x = x.float().reshape((x.shape[0], -1))
            pred = self.output_layer(x)
            pred_tree = u.njtree(pred, namespace=true_tree.taxon_namespace)
            d += u.rf_distance(true_tree, pred_tree)
            n += 1
        return d/n


class Transpose(Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class Lambda(Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class EmbedLayer(Module):
    """implements dropout for arbitrary dtype"""

    def __init__(self, embed_dim: int, dropout: int = 0, device='cpu', channels_last=False):
        super().__init__()
        self.p = dropout
        self.add_module('embed', torch.nn.Embedding(
            num_embeddings=N_STATES,
            embedding_dim=embed_dim))
        self.channels_last = channels_last
        # self.to(device)
        # self.device = device

    def dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.p > 0:
            mask = torch.rand_like(x, dtype=torch.float16) > self.p
            return x.mul(mask)
        return x

    def forward(self, x):
        x = self.dropout(x).long()
        x = self.embed(x)
        x = self.dropout(x)
        x = x.transpose(-2, -1)
        return(x)

    # def to(self, device, **kwargs):
    #     super().to(device=device, **kwargs)
    #     self.device = device


class CovarianceDecoder(Module):
    r"""returns inner product of embeddings."""

    def __init__(self, clip=1e5, as_list=False):
        super().__init__()
        self.clip = clip
        self.as_list = as_list

    def __repr__(self):
        return f'CovarianceDecoder(as_list={self.as_list}, clip={self.clip})'

    def forward(self, z, batches):
        r"""
        Input: a sequence of pxn matrices of n-dimensional embeddings.
        Output: a list of pxp gram matrices.
        """
        z = torch.clamp(z, min=-self.clip, max=self.clip)
        if batches is None:
            n = z.size(0)
            pd = torch.pdist(z)
            return squareform(pd, n)
        preds = []
        for s in batch_iter(z, batches, square=False):
            C = torch.mm(s, s.t())
            preds.append(C)
        if self.as_list:
            return preds
        return torch.block_diag(*preds)


class MetricDecoder(Module):
    """returns pairwise distance matrix over input embeddings.
    These are calculated independently for each item in the batch, so
    """

    def __init__(self, clip=1e5, as_list=False):
        super().__init__()
        self.clip = clip
        self.as_list = as_list

    def forward(self, z, batches=None):
        z = torch.clamp(z, min=-self.clip, max=self.clip)
        if z.dim() == 2:
            if batches is None:
                n = z.size(0)
                pd = torch.pdist(z)
                return squareform(pd, n)
            generator = batch_iter(z, batches, square=False)
        else:
            generator = (x for x in z)
        preds = []
        for s in generator:
            n = s.size(0)
            pd = torch.pdist(s)
            if self.as_list:
                preds.append(pd)
            else:
                preds.append(squareform(pd, n))
        if self.as_list:
            return preds
        return torch.block_diag(*preds)


def make_conv_net_2d(conv_layer_sizes: List,
                     kernel: int,
                     stride: int,
                     dropout: float,
                     nonlinearity=nn.ELU,
                     batch_norm=False):
    """Assumes that the first spatial dim is the number of taxa; convolves in the second spatial dim only

    Args:
        conv_layer_sizes (list): channels
        kernel (int): same kernel size for all layers
        stride (int): stride
        dropout (float): dropout
        nonlinearity (callable, optional): nonlinear activation. Defaults to nn.ELU.
        batch_norm (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    padding = 'same' if stride == 1 else 'valid'
    conv_layers = nn.ModuleList()
    for in_channels, out_channels in zip(conv_layer_sizes, conv_layer_sizes[1:]):
        conv_layers.append(nonlinearity())
        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.extend([
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(1, kernel),
                      stride=(1, stride),
                      padding=padding,
                      bias=not batch_norm),
            nn.Dropout(p=dropout)]
        )
    return nn.Sequential(*conv_layers)


def make_conv_net(conv_layer_sizes,
                  kernel,
                  stride,
                  dropout,
                  nonlinearity=nn.ELU,
                  batch_norm=False):
    padding = 'same' if stride == 1 else 'valid'
    conv_layers = nn.ModuleList()
    for in_channels, out_channels in zip(conv_layer_sizes, conv_layer_sizes[1:]):
        conv_layers.append(nonlinearity())
        if batch_norm:
            conv_layers.append(nn.BatchNorm1d(in_channels))
        conv_layers.extend([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel,
                      stride=stride,
                      padding=padding,
                      bias=not batch_norm),
            nn.Dropout(p=dropout)]
        )
    return nn.Sequential(*conv_layers)


def build_fc_network(in_channels: int = None,
                     out_channels: int = None,
                     layers: list[int] = None,
                     batch_norm=True,
                     layer_norm=False,
                     nonlinearity: nn.Module = nn.ReLU):
    """Adds nonlinearity *before* each layer.

    Args:
        in_channels (int, optional): single layer. Defaults to None.
        out_channels (int, optional): single layer. Defaults to None.
        layers (list[int], optional): if specified, will generate multilayer network. Defaults to None.
        batch_norm (bool, optional): add batch norm BEFORE each layer. Defaults to True.
        nonlinearity (nn.Module, optional): callable. Defaults to nn.ReLU.

    Returns:
        _type_: _description_
    """
    modules = nn.ModuleList([])
    bias = not batch_norm
    if layers is not None:
        if in_channels is not None:
            layers.insert(0, in_channels)
        if out_channels is not None:
            layers.append(out_channels)
        for in_channels, out_channels in zip(layers, layers[1:]):
            if nonlinearity is not None:
                modules.append(nonlinearity())
            if batch_norm:
                modules.append(nn.BatchNorm1d(in_channels))
            if layer_norm:
                modules.append(nn.LayerNorm(in_channels))
            modules.append(nn.Linear(in_channels, out_channels, bias=bias))
    else:
        if nonlinearity is not None:
            modules.append(nonlinearity())
        if batch_norm:
            modules.append(nn.BatchNorm1d(in_channels))
        if layer_norm:
                modules.append(nn.LayerNorm(in_channels))
        modules.append(nn.Linear(in_channels, out_channels, bias=bias))
    return nn.Sequential(*modules)
