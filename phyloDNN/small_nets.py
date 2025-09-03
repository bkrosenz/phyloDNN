from sqlite3 import adapt
from pandas import infer_freq
from sympy import E
import torch.nn.functional as F
from einops import rearrange
from pathlib import Path
from typing import Literal, List, Sequence
from torch import adaptive_avg_pool1d, layer_norm, nn
import math
from functools import lru_cache
import torch
from scipy.special import binom

# from phyloDNN.models.graph_utils import build_fc_network

MAX_GENETIC_DIST = 9.0


def value_counts(x):
    L = x.shape[-1]
    m = torch.tensor(
        [
            [(torch.bincount(xx) ** 2).sum() for xx in batch.long()]
            for batch in x.squeeze(1)
        ],
        device=x.device,
    ).float()
    return 1 - m / L**2


def build_fc_network(n_layers: int, fc1: int, fc2: int = None, activation=nn.ELU):
    """Output dim is 1"""
    fc_layers = nn.ModuleList()
    if fc2 is None:
        fc2 = fc1
    # fc1,fc2,*layer_dims =layer_dims

    for i in range(n_layers):
        fc_layers.extend(
            [
                nn.Linear(fc1, fc2),
                activation(),
            ]
        )
        fc1 = fc2
        fc2 = max(fc2 // 2, 1)
    fc_layers.append(nn.Linear(fc1, 1))
    return nn.Sequential(*fc_layers)


class PermutationEquivariantLayer(nn.Module):
    """computes \lambda I + \mu 11^T  output has same shape as input.
    where sum is over pairs/taxa dimension rather than sites"""

    def __init__(
        self,
        invariant=False,
        heads: int = 1,
        axial: bool = False,
        dimension: str = "taxa",
    ):
        """Heads parameter allows subsets of channels to be grouped together.
        Dont need this to have an invariant version except for very simple distance functions.
        If dimension is "both", then the output is invariant to both taxa and sites, i.e axial.
        """
        super().__init__()
        self.heads = heads
        self.invariant = invariant
        self.lam = nn.Linear(heads, heads)
        self.mu = nn.Linear(heads, heads)
        self.axial = axial
        if dimension == "taxa":
            self.dimension = -3
        elif dimension == "sites":
            self.dimension = -2
        elif dimension == "both":
            self.dimension = -2
            # nu will take the taxa dimension
            self.nu = nn.Linear(heads, heads)
        else:
            raise ValueError("unknown dimension")

    def forward(self, x: torch.Tensor):
        """batches x channels x n_taxa x n_sites -> batches x channels/h x n_taxa x n_sites x heads"""
        x = rearrange(x, "b (c h) p l -> b c p l h", h=self.heads)
        out = self.mu(x.sum(self.dimension, keepdim=True))
        if hasattr(self, "nu"):
            # expands back to L dimension
            out = out + self.nu(x.sum(-3, keepdim=True))
        if not self.invariant:
            out = out.expand(x.shape) + self.lam(x)
        return rearrange(out, "b c p l h -> b (c h) p l")


class PairEquivariantLayer(nn.Module):
    """computes \alpha I + \beta [0, I; I, 0] + \delta [0,11^T;11^T,0] + \gamma [11^T,0;0,11^T] , output has same shape as input.
    If dimension is 'both', will in addition sum across the taxa dimension."""

    def __init__(
        self,
        invariant: bool = False,
        heads: int = 1,
        dimension="sites",
    ):
        """Heads parameter allows subsets of channels to be grouped together."""
        super().__init__()

        self.invariant = invariant
        self.heads = heads
        self.alpha = nn.Linear(heads, heads)
        if not invariant:
            self.beta = nn.Linear(heads, heads)
            self.gamma = nn.Linear(heads, heads)
            self.delta = nn.Linear(heads, heads)
        if dimension == "both":
            #     if invariant:
            #         raise ValueError("cannot have taxa (col) equivariant layer that is also site-invariant")
            if not invariant:
                self.epsilon = nn.Linear(heads, heads)
            self.zeta = nn.Linear(heads, heads)  # taxa invariant

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """batches x channels x n_pairs x n_sites"""
        x, y = map(
            lambda z: rearrange(z, "b (c h) ... -> b c ... h", h=self.heads), (x, y)
        )  # batches x channels x n_pairs x n_sites x heads
        x_sum = x.sum(dim=-2, keepdim=True)  # sum across sites
        y_sum = y.sum(dim=-2, keepdim=True)

        if self.invariant:
            out = self.alpha(x_sum + y_sum)
            if hasattr(self, "zeta"):  # batches x channels x taxa x 1 x heads
                out = out + self.zeta(
                    y_sum.sum(2, keepdim=True) + x_sum.sum(2, keepdim=True)
                )  # sum again over taxa -> batches x channels x 1 x 1 x heads
            return rearrange(out, "b c ... h -> b (c h) ...", h=self.heads).squeeze(
                dim=-1
            )
        else:
            x_out = self.alpha(x) + self.beta(y) + self.gamma(x_sum) + self.delta(y_sum)
            y_out = self.alpha(y) + self.beta(x) + self.gamma(y_sum) + self.delta(x_sum)
            if hasattr(self, "zeta"):
                x_col_sum = x.sum(dim=-3, keepdim=True)
                y_col_sum = y.sum(dim=-3, keepdim=True)
                x_out = x_out + self.epsilon(x_col_sum) + self.zeta(y_col_sum)
                y_out = y_out + self.epsilon(y_col_sum) + self.zeta(x_col_sum)
            x, y = map(
                lambda z: rearrange(z, "b c ... h -> b (c h) ...", h=self.heads),
                (x_out, y_out),
            )

            return x, y


class Stack(nn.Module):
    """fNNs include activation layer"""

    def __init__(
        self,
        hidden_channels,
        n_layers,
        activation,
        invariant,
        n_fnn_layers: int = 2,
        conv_layers: List | None = None,
        stride=1,
        **kwargs,
    ):
        super().__init__()
        self.invariant = invariant
        if activation == "elu":
            activation_layer = nn.ELU
            self.activation = F.elu
        elif activation == "relu":
            activation_layer = nn.ReLU
            self.activation = F.relu
        elif activation == "gelu":
            activation_layer = nn.GELU
            self.activation = F.gelu
        else:
            raise ValueError(f"activation {activation} not supported")
        self.hidden_channels = hidden_channels
        self.fNNs = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(n_layers):
            modules = []
            for _ in range(n_fnn_layers):
                modules.extend(
                    [
                        nn.Conv2d(
                            in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            kernel_size=1,
                            stride=1,
                        ),
                        activation_layer(),
                    ]
                )
            self.fNNs.append(nn.Sequential(*modules))
            self.layernorms.append(nn.LayerNorm(hidden_channels))

        if conv_layers:
            layers = []
            for d in conv_layers:
                layers.extend(
                    [
                        nn.Conv1d(
                            hidden_channels,
                            hidden_channels,
                            kernel_size=d,
                            stride=stride,
                        ),
                        activation_layer(),
                    ]
                )
            self.conv_layer = nn.Sequential(*layers)

    def conv(self, x):
        """batches x channels x n_pairs x n_sites"""
        batch_size = x.shape[0]
        x = rearrange(x, "b c p l -> (b p) c l")
        x = self.conv_layer(x)
        x = rearrange(x, "(b p) c l -> b c p l", b=batch_size)
        return x


# compilation slows down training by 3x on A100 and 10x on V100
# @torch.compile(dynamic=True)
class SequenceEquivariantStack(Stack):

    def __init__(
        self,
        hidden_channels,
        n_heads=4,
        n_layers=6,
        invariant=False,
        activation="elu",
        dimension="sites",
        conv_layers: List | None = None,
        stride: int = 2,
        residual=False,
        n_fnn_layers: int = 2,
        **kwargs,
    ):
        """If n_layers==0 and invariant==False this is just an identity function"""
        super().__init__(
            n_layers=n_layers,
            hidden_channels=hidden_channels,
            activation=activation,
            invariant=invariant,
            conv_layers=conv_layers,
            stride=stride,
            n_fnn_layers=n_fnn_layers,
        )

        # TODO: should the sum over sites be normalized by # sites? can this be solved by using > 1 heads, so one feature can be all ones (would need to train on aligns of different lengths)
        self.equivariant_layers = nn.ModuleList()

        for i in range(n_layers):
            self.equivariant_layers.append(
                PermutationEquivariantLayer(
                    heads=n_heads, dimension=dimension, invariant=False
                )
            )
        if invariant:
            # last layer can only be over sites, since we collapse this dim
            self.equivariant_layers.append(
                PermutationEquivariantLayer(
                    heads=n_heads, dimension="sites", invariant=True
                )
            )
        self.invariant = invariant
        self.residual = residual

    def forward(self, x):
        """input must be shape batches x channels x n_taxa x n_sites"""

        # can't do residual connections with conv layer
        if hasattr(self, "conv_layer"):
            x = self.conv(x)

        for i in range(self.n_layers):
            xr = x if self.residual else 0.0
            x = self.activation(self.equivariant_layers[i](x))
            x = self.fNNs[i](x)
            x = x + xr  # residual connection
            x = self.layernorms[i](x.transpose(-1, -3)).transpose(-1, -3)

        if self.invariant:
            return self.equivariant_layers[-1](x)
        return x


class PairEquivariantStack(Stack):

    def __init__(
        self,
        hidden_channels,
        n_heads=4,
        n_layers=6,
        invariant=True,
        dimension="sites",
        activation: str = "elu",
        conv_layers: List | None = None,
        stride: int = 2,
        n_fnn_layers: int = 2,
        **kwargs,
    ):
        """Pair equivariant stack with convolutional layers.
        Args:
            hidden_channels: number of channels
            n_heads: number of heads
            n_layers: number of layers
            invariant: if True, the last layer is invariant.
            axial: if True, interleave site and taxa equivariant layers.
            activation: activation function. one of "elu", "relu", "gelu".
            conv_layers: list of convolutional layer dims
            stride: stride for convolutional layers. should be > 1. default is 2.
        """
        super().__init__(
            n_layers=n_layers,
            hidden_channels=hidden_channels,
            activation=activation,
            invariant=invariant,
            conv_layers=conv_layers,
            stride=stride,
            n_fnn_layers=n_fnn_layers,
        )
        self.equivariant_layers = nn.ModuleList()

        for i in range(n_layers):
            self.equivariant_layers.append(
                PairEquivariantLayer(
                    heads=n_heads, dimension=dimension, invariant=False
                )
            )
        self.invariant_layer = PairEquivariantLayer(
            heads=n_heads, dimension=dimension, invariant=True
        )

    @classmethod
    @lru_cache
    def seq2pair(
        self, nb_seq: int, device: torch.device | str | None = None
    ) -> torch.Tensor:
        # no_diagonal: bool = True
        """creates indexer to transform seqs to seq pairs.

        Args:
            nb_seq (_type_): number of seqs
        Returns:
            _type_: _description_
        """
        nb_pairs = int(binom(nb_seq, 2))

        ix = torch.zeros(2, nb_pairs, device=device, dtype=int)
        k = 0
        for i in range(nb_seq):
            for j in range(i + 1, nb_seq):
                ix[0, k] = i
                ix[1, k] = j
                k = k + 1

        return ix

    @classmethod
    def get_pairs(self, x: torch.Tensor):
        """output is dimensions batches x hidden_channels x n_pairs x n_sites."""
        idx = self.seq2pair(x.shape[-2], x.device)
        x, y = x[..., idx[0], :], x[..., idx[1], :]

        return x, y

    def forward(self, x):
        """input must be shape batches x channels x n_pairs x 2*n_sites"""
        x, y = self.get_pairs(x)  # batches x hidden_channels x npairs x n_sites
        for i in range(self.n_layers):
            x_new, y_new = map(self.activation, self.equivariant_layers[i](x, y))

            x, y = (
                self.fNNs[i](x_new) + x,
                self.fNNs[i](y_new) + y,
            )  # fNN includes activation

            x, y = (
                self.layernorms[i](m.transpose(-1, -3)).transpose(-1, -3)
                for m in (x, y)
            )
            # y = self.layernorms[i](y.transpose(-1, -3)).transpose(-1, -3)  #

        # can't do residual connections with conv layer
        if hasattr(self, "conv_layer"):
            x, y = self.conv(x), self.conv(x)

        # if hasattr(self, "invariant_layer"):
        return self.invariant_layer(x, y)
        # return x, y


class CondensedPairEquivariantStack(SequenceEquivariantStack):

    @classmethod
    @lru_cache
    def seq2pair(
        self, nb_seq: int, device: torch.device | str | None = None
    ) -> torch.Tensor:
        # no_diagonal: bool = True
        """creates indexer to transform seqs to seq pairs.

        Args:
            nb_seq (_type_): number of seqs
        Returns:
            _type_: _description_
        """
        nb_pairs = int(binom(nb_seq, 2))

        ix = torch.zeros(2, nb_pairs, device=device, dtype=int)
        k = 0
        for i in range(nb_seq):
            for j in range(i + 1, nb_seq):
                ix[0, k] = i
                ix[1, k] = j
                k = k + 1

        return ix

    @classmethod
    def get_pairs(self, x: torch.Tensor):
        """output is dimensions batches x hidden_channels x n_pairs x n_sites."""
        idx = self.seq2pair(x.shape[-2], x.device)
        x, y = x[..., idx[0], :], x[..., idx[1], :]

        return x + y

    def forward(self, x):
        """input must be shape batches x channels x n_pairs x 2*n_sites"""
        x = self.get_pairs(x)  # batches x hidden_channels x npairs x n_sites
        for i in range(self.n_layers):
            x = self.activation(self.equivariant_layers[i](x))
            x = self.fNNs[i](x)
            x = self.layernorms[i](x.transpose(-1, -3)).transpose(-1, -3)

        if self.invariant:
            return self.equivariant_layers[-1](x)

        # if hasattr(self, "invariant_layer"):
        return x


class SmallNet(nn.Module):

    def __init__(
        self,
        hidden_channels=128,
        n_heads=4,
        n_fc_layers=3,
        n_seq_layers=6,
        n_pair_layers=6,
        activation="elu",
        dimension="sites",
        conv_layers=None,
        fc_channels=None,
        n_intermediate_layers: int = 0,
        intermediate_channels: List | None = None,
        n_fnn_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.n_seq_layers = n_seq_layers
        self.n_pair_layers = n_pair_layers
        self.activation = activation
        self.embed = nn.Embedding(num_embeddings=22, embedding_dim=hidden_channels)
        self.seqnet = SequenceEquivariantStack(
            hidden_channels=hidden_channels,
            n_heads=n_heads,
            n_layers=n_seq_layers,
            activation=activation,
            n_fnn_layers=n_fnn_layers,
            conv_layers=conv_layers,
            dimension=dimension,
            invariant=False,
            adaptive_pool=False,
        )
        # TODO add vanilla Stack option for no equivariant layers
        if n_pair_layers > 0:
            pairnet = PairEquivariantStack(
                hidden_channels=hidden_channels,
                n_heads=n_heads,
                n_layers=n_pair_layers,
                dimension=dimension,
                activation=activation,
                n_fnn_layers=n_fnn_layers,
                conv_layers=conv_layers,
            )
            self.get_pair_distances = pairnet
        # This gets shadowed by torch.pdist for seqnets.
        else:
            self.get_pair_distances = torch.vmap(torch.vmap(torch.pdist))

        # TODO add adaptive pooling for before fully connected layers
        # if adaptive_pool:
        #     self.pooling = nn.Sequential(
        #         nn.AdaptiveAvgPool1d((hidden_channels,128)),

        if n_intermediate_layers > 0 or intermediate_channels is not None:
            self.fNNs = nn.ModuleList()
            self.layernorms = nn.ModuleList()
            if intermediate_channels is None:
                intermediate_channels = [hidden_channels] * (n_intermediate_layers + 1)
            else:
                intermediate_channels = [hidden_channels] + intermediate_channels
            for f_in, f_out in zip(
                intermediate_channels[:-1], intermediate_channels[1:]
            ):
                self.fNNs.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=f_in,
                            out_channels=f_out,
                            kernel_size=1,
                            stride=1,
                        ),
                        nn.ELU(),
                    )
                )
                self.layernorms.append(nn.LayerNorm(f_out))
            hidden_channels = intermediate_channels[-1]

        if fc_channels is None:
            fc_channels = hidden_channels
        self.fc_layer = build_fc_network(n_fc_layers, hidden_channels, fc_channels)

    @classmethod
    @torch.compiler.disable()
    @lru_cache
    def seq2pair(
        self, nb_seq: int, device: torch.device | str | None = None
    ) -> torch.Tensor:
        # no_diagonal: bool = True
        """creates indexer to transform seqs to seq pairs.

        Args:
            nb_seq (_type_): number of seqs
        Returns:
            _type_: _description_
        """
        nb_pairs = int(binom(nb_seq, 2))

        ix = torch.zeros(2, nb_pairs, device=device, dtype=int)
        k = 0
        for i in range(nb_seq):
            for j in range(i + 1, nb_seq):
                ix[0, k] = i
                ix[1, k] = j
                k = k + 1

        return ix

    @classmethod
    def get_pairs(self, x: torch.Tensor):
        """output is dimensions batches x hidden_channels x n_pairs x n_sites."""
        idx = self.seq2pair(x.shape[-2], x.device)
        x, y = x[..., idx[0], :], x[..., idx[1], :]

        return x, y

    def embed_transform(self, x: torch.Tensor):
        """x: batches x hidden_channels x n_seq x 1"""
        """transform the compressed sequences before computing pairwise distances."""
        for fnn, layer_norm in zip(self.fNNs, self.layernorms):
            x = fnn(x)
            x = layer_norm(x.transpose(-1, -3)).transpose(-1, -3)
        return x

    def forward(
        self,
        x: torch.Tensor,
    ):
        """x: batches x n_seq x n_sites"""
        x = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )  # batches x hidden_channels x n_seq x n_sites

        x = self.seqnet(x) + (
            x if hasattr(self, "pairnet") else 0
        )  # residual connection

        if hasattr(self, "fNNs"):
            # batches x hidden_channels x n_seq x 1
            out = self.embed_transform(x)
            if out.shape == x.shape:  # residual
                x = x + out
            else:
                x = out

        x = self.get_pair_distances(x.squeeze(-1))
        x = self.fc_layer(x.squeeze(dim=-1).transpose(-1, -2))
        return x.squeeze(dim=-1)


class BourgainNet(SmallNet):

    def __init__(
        self,
        hidden_channels=32,
        n_heads=4,
        n_seq_layers=6,
        activation="elu",
        dimension="sites",
        n_intermediate_layers: int = 4,
        n_fnn_layers: int = 2,
        fc_channels=32,
        n_fc_layers: int = 0,
        **kwargs,
    ):
        super().__init__(
            n_fc_layers=0,
            n_heads=1,
            hidden_channels=hidden_channels,
            activation="elu",
            fc_channels=0,
            n_seq_layers=0,
            n_pair_layers=0,
        )
        self.seqnet = SequenceEquivariantStack(
            hidden_channels=hidden_channels,
            n_heads=n_heads,
            n_layers=n_seq_layers,
            activation=activation,
            n_fnn_layers=n_fnn_layers,
            dimension=dimension,
            invariant=True,
            adaptive_pool=False,
            residual=True,
        )
        self.get_pair_distances = torch.vmap(torch.pdist)

        # TODO replace this with a dim=taxa equivariant stack
        self.fNNs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ELU(),
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=fc_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                )
            ]
        )
        self.layernorms = nn.ModuleList()
        for _ in range(n_intermediate_layers):
            self.fNNs.append(
                nn.Sequential(
                    nn.ELU(),
                    nn.Conv2d(
                        in_channels=fc_channels,
                        out_channels=fc_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                )
            )
            self.layernorms.append(nn.LayerNorm(fc_channels))
        if n_fc_layers > 0:
            self.fc_layer = build_fc_network(n_fc_layers, 1, fc_channels)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """x: batches x n_seq x n_sites"""
        x = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )  # batches x hidden_channels x n_seq x n_sites
        x = self.seqnet(x)  # batches x hidden_channels x n_seq x 1

        x = self.embed_transform(x)  # batches x fc_channels x n_seq x 1

        x = self.get_pair_distances(x.squeeze(-1).transpose(-1, -2)).unsqueeze(
            -1
        )  # Euclidean
        # batches x n_pairs
        if hasattr(self, "fc_layer"):
            x = self.fc_layer(x)

        return x.squeeze(dim=-1)  # batches x n_pairs


class K2PNetExact(SmallNet):

    def __init__(self, **kwargs):
        super().__init__(
            n_fc_layers=0,
            n_heads=1,
            hidden_channels=0,
            activation="elu",
            fc_channels=0,
            n_seq_layers=0,
            n_pair_layers=0,
        )
        self.ratio = 2  # ratio of transitions to transversions

    def forward(self, x: torch.Tensor):
        x = x.long().unsqueeze(1)  # batches x 1 x n_seq x n_sites
        x, y = self.get_pairs(x.squeeze(-1))
        transitions = (
            (x == 2) & (y == 7)
            | (x == 7) & (y == 2)
            | (x == 3) & (y == 18)
            | (x == 18) & (y == 3)
        )
        transversions = (x != y) & ~transitions

        # AG,CT

        # transversions = (
        #     x
        #     == 2 & y
        #     == 3 | x
        #     == 3 & y
        #     == 2 | x
        #     == 7 & y
        #     == 18 | x
        #     == 18 & y
        #     == 7 | x
        #     == 2 & y
        #     == 18 | x
        #     == 18 & y
        #     == 2 | x
        #     == 3 & y
        #     == 7 | x
        #     == 7 & y
        #     == 3
        # ).float().mean()
        # AT,AC,GT,GC
        x = torch.stack(
            [transitions.float().mean(-1), transversions.float().mean(-1)], dim=0
        )
        x = -(1 - 2 * x[0] - x[1]).log() / 2 - (1 - 2 * x[1]).log() / 4
        # x = self.fc_layer(x.squeeze(dim=-1).transpose(-1, -2))
        return torch.nan_to_num(
            x.squeeze(dim=-2), nan=MAX_GENETIC_DIST, posinf=MAX_GENETIC_DIST
        )


class HammingExact(SmallNet):
    """Hamming distance."""

    def __init__(self, **kwargs):
        super().__init__(
            n_fc_layers=0,
            n_heads=1,
            hidden_channels=0,
            fc_channels=0,
            n_seq_layers=0,
            n_pair_layers=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        """x: batches x n_seq x n_sites"""
        x = x.long().unsqueeze(1)  # batches x hidden_channels x n_seq x n_sites
        x, y = self.get_pairs(x)
        return torch.mean((x != y).float(), -1).squeeze(dim=-2)


class K2PNet(SmallNet):
    """ACGT = 2,3,7,18.  K2P model with trainable network to learn p,p \mapsto -ln(1-2p-q)/2 -ln(1-2q)/4 where p is the transition rate and q is the transversion rate."""

    def __init__(self, n_fc_layers=3, activation="elu", fc_channels=None, **kwargs):
        super().__init__(
            n_fc_layers=n_fc_layers,
            n_heads=1,
            hidden_channels=2,
            activation=activation,
            fc_channels=fc_channels,
            n_seq_layers=0,
            n_pair_layers=0,
        )
        self.ratio = 2  # ratio of transitions to transversions
        del self.embed

    def forward(self, x: torch.Tensor):
        x = x.long().unsqueeze(1)  # batches x 1 x n_seq x n_sites
        x, y = self.get_pairs(x.squeeze(-1))
        transitions = (
            (x == 2) & (y == 7)
            | (x == 7) & (y == 2)
            | (x == 3) & (y == 18)
            | (x == 18) & (y == 3)
        )  # AG,CT
        transversions = (x != y) & ~transitions  # AT,AC,GT,GC
        x = torch.stack(
            [transitions.float().mean(-1), transversions.float().mean(-1)], dim=-1
        ).squeeze(
            dim=1
        )  # batches x n_pairs
        x = torch.nan_to_num(x)
        x = self.fc_layer(x)
        return x.squeeze(dim=-1)


class JCNetExact(SmallNet):
    """Jukes-Cantor model, ACGT = 2,3,7,18.  Trainable transformation of Hamming distance."""

    def __init__(self, dna=True, maclaurin=False, f81=False, **kwargs):
        super().__init__(
            n_fc_layers=0,
            n_heads=1,
            hidden_channels=0,
            fc_channels=0,
            n_seq_layers=0,
            n_pair_layers=0,
        )
        self.maclaurin=maclaurin
        self.dna = dna
        self.f81 = f81
        self.bins = torch.Tensor([2.0, 3.0, 7.0, 18.0]) + 0.5
        # self.freq = torch.vmap(value_counts)

    def forward(
        self,
        x: torch.Tensor,
    ):
        """x: batches x n_seq x n_sites"""
        # x = self.embed(x.long())

        x = x.long().unsqueeze(1)  # batches x hidden_channels x n_seq x n_sites
        x, y = self.get_pairs(x)
        if self.f81:
            b = value_counts(torch.cat([x, y], -1))
        x = torch.mean((x != y).float(), -1)
        if self.f81:
            x = -b * (1 - x / b).log()
        elif self.dna:
            x = 4 * x / 3
            if self.maclaurin:
                x = 3 * (x + x**2 / 2 + x**3 / 3 + x**4 / 4 + x**5 / 5) / 4
            else:
                x = -3 * (1 - x).log() / 4  # JC distance
        else:  # AA sequence
            x = -21 * (1 - 22 * x / 21).log() / 22

        return torch.nan_to_num(x.squeeze(dim=-2), nan=MAX_GENETIC_DIST, posinf=MAX_GENETIC_DIST)

class JCNet(SmallNet):
    """Jukes-Cantor model, ACGT = 2,3,7,18.  Trainable transformation of Hamming distance."""

    def __init__(self, n_fc_layers=3, activation="elu", fc_channels=1, **kwargs):
        super().__init__(
            n_fc_layers=n_fc_layers,
            n_heads=1,
            hidden_channels=1,
            activation=activation,
            fc_channels=fc_channels,
            n_seq_layers=0,
            n_pair_layers=0,
        )
        del self.embed

    def forward(
        self,
        x: torch.Tensor,
    ):
        """x: batches x n_seq x n_sites"""
        # x = self.embed(x.long())
        x = x.long().unsqueeze(1)  # batches x hidden_channels x n_seq x n_sites
        x, y = self.get_pairs(x)
        x = torch.mean((x != y).float(), -1)
        x = self.fc_layer(x.squeeze(dim=-1).transpose(-1, -2))
        return x.squeeze(dim=-1)


class GTRNet(SmallNet):
    """Jukes-Cantor model, ACGT = 2,3,7,18.  Trainable transformation of Hamming distance."""

    def __init__(self, n_fc_layers=3, activation="elu", fc_channels=1, **kwargs):
        super().__init__(
            n_fc_layers=n_fc_layers,
            n_heads=1,
            hidden_channels=2,
            activation=activation,
            fc_channels=fc_channels,
            n_seq_layers=0,
            n_pair_layers=0,
        )
        del self.embed

    def forward(
        self,
        x: torch.Tensor,
    ):
        """x: batches x n_seq x n_sites"""
        # x = self.embed(x.long())
        x = x.long().unsqueeze(1)  # batches x 1 x n_seq x n_sites
        x, y = self.get_pairs(x)

        x = torch.cat((x, y), 1).float()  # batches x 2 x n_pairs x n_sites
        x = self.fc_layer(x.squeeze(dim=-1).transpose(-1, -2))
        # TODO: turn this into \frac{\sum_{i=1}^NS[x_i,y_i]}{ \max\left(\sum_{i=1}^NS[x_i,x_i] ,\sum_{i=1}^NS[y_i,y_i]

        return x.mean(-1).squeeze(dim=-1)
