from pandas import infer_freq
from sympy import E
import torch.nn.functional as F
from einops import rearrange
from pathlib import Path
from typing import Literal, List, Sequence
from torch import nn
import math
from functools import lru_cache
import torch
from scipy.special import binom

from phyloDNN.models.graph_utils import build_fc_network
from .models.layers import (
    AxialAttention,
    CompressedMultiAttention,
    EquivariantLayer,
    KernelAxialMultiAttention,
    LinearMultiAttention,
    PermutationEquivariantLayer,
    PositionalEncoding,
)

class Preprocessor(nn.Module):
    """always on CPU"""

    def __init__(self, network: nn.Module | Path, device) -> None:
        super().__init__()
        if isinstance(network, Path):
            network = torch.load(network, map_location=device)
        for m in tuple(network._modules.keys()):
            if m not in ("embed", "norm", "pos_encoding", "block_1_1"):
                del network._modules[m]
        for p in network.parameters():
            p.requires_grad = False
        self.network = network

    def forward(self, x):
        return self.network.compute_embeddings(x)


class AttentionStack(nn.Module):
    def __init__(
        self,
        hidden_channels,
        downsample="every_layer",
        downsample_strategy="avg",
        positional: bool = False,
        attention_heads=4,
        n_attention_layers=6,
        compressed_row_sizes: List | None = None,
        field_size_col: int | None = None,
        field_size_row: int | None = None,
        causal=False,
        kernel_size=10,
        with_q=False,
        sdpa: bool = False,
        share_qk=False,
        unfold: bool = False,
        local: bool = False,
        stride: int = 1,
        row: bool = True,
        col: bool = True,
        **kwargs,
    ):
        super().__init__()
        if compressed_row_sizes is None:
            self.n_blocks = n_attention_layers
            self.compressed = False
        else:
            self.n_blocks = len(compressed_row_sizes)
            self.compressed = True

        downsampling_layers = nn.ModuleList()
        row_attentions = nn.ModuleList()
        column_attentions = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.fNNs = nn.ModuleList()
        fNN_channels = attention_heads * hidden_channels
        for i in range(self.n_blocks):
            if row:
                if compressed_row_sizes and compressed_row_sizes[i] is not None:
                    row_block = LinearMultiAttention(
                        hidden_channels,
                        attention_heads,
                        pos_embed=positional,
                        sdpa=sdpa,
                        n_output=compressed_row_sizes[i],
                    )
                # elif local:
                #     row_block = LocalAttentionBlock(
                #         hidden_channels,
                #         attention_heads,
                #         share_qk=share_qk,
                #         pos_embed=positional,
                #         field_size=field_size_row,
                #     )
                else:
                    if field_size_row is not None:
                        raise NotImplementedError("field_size_row requires local==True")
                    row_block = KernelAxialMultiAttention(
                        hidden_channels,
                        attention_heads,
                        pos_embed=positional,
                        share_qk=share_qk,
                        field_size=field_size_row,
                        unfold=unfold,
                        with_q=with_q,
                    )
                row_attentions.append(row_block)
            if col:

                # else:
                column_attentions.append(
                    KernelAxialMultiAttention(
                        hidden_channels,
                        attention_heads,
                        share_qk=share_qk,
                        field_size=field_size_col,
                        with_q=with_q,
                    )
                )
            self.layernorms.append(nn.LayerNorm(hidden_channels))
            self.fNNs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=fNN_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        in_channels=fNN_channels,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                )
            )
            self.downsample_every_layer = downsample == "every_layer"
            # downsample every other layer
            if downsample_strategy == "avg":
                pool = nn.AvgPool2d(kernel_size=(1, kernel_size), stride=(1, stride))
            elif downsample_strategy == "max":
                pool = nn.MaxPool2d(kernel_size=(1, kernel_size), stride=(1, stride))
            if stride > 1 and (self.downsample_every_layer or not i % 2):
                if downsample_strategy == "conv":
                    downsampling_layers.append(
                        nn.Sequential(
                            nn.GELU(),
                            nn.Conv2d(
                                in_channels=hidden_channels,
                                out_channels=fNN_channels,
                                kernel_size=(1, kernel_size),
                                stride=(1, stride),
                            ),
                            nn.GELU(),
                            nn.Conv2d(
                                in_channels=fNN_channels,
                                out_channels=hidden_channels,
                                kernel_size=1,
                                stride=1,
                            ),
                        )
                    )
                else:
                    downsampling_layers.append(pool)
        if row:
            self.row_attentions = row_attentions
        if col:
            self.column_attentions = column_attentions

        if stride > 1:
            self.downsampling_layers = downsampling_layers

    def forward(self, out, col_mask=None, intermediate=False, debug=False):
        if intermediate:
            output = []
        # if debug:
        #     print("before attn", out[0, :, 0, :].unique(dim=-1).shape)
        for i in range(self.n_blocks):
            # AXIAL ATTENTIONS BLOCK
            # ----------------------
            # ROW ATTENTION
            if hasattr(self, "row_attentions"):
                # (batch_size,features,nb_pairs,seq_len) -> (batch_size,nb_pairs,seq_len,features)
                att, _ = self.row_attentions[i](out.permute(0, 2, 3, 1))
                # row attention+residual connection
                out = att.permute(0, 3, 1, 2) + (0 if self.compressed else out)
                out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                    -1, -3
                )  # layernorm
                # if debug:
                #     print(i, "row", out[0, :, 0, :].unique(dim=-1).shape)

            # COLUMN ATTENTION
            if hasattr(self, "column_attentions"):
                # (batch_size,features,nb_pairs,seq_len) -> (batch_size,seq_len,nb_pairs,features)
                att, _ = self.column_attentions[i](out.permute(0, 3, 2, 1), col_mask)

                # column attention+residual connection
                out = att.permute(0, 3, 2, 1) + out
                out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                    -1, -3
                )  # layernorm
                # if debug:
                #     print(i, "col", out[0, :, 0, :].unique(dim=-1).shape)
            # if debug:
            #     final_pattern_count = out[0, :, 0, :].unique(dim=-1).shape
            if intermediate:
                output.append(
                    self.compute_distances(out.detach().permute(0, 2, 3, 1).flatten(-2))
                )
            # FEEDFORWARD+
            # TODO: can we add conv/pooling here without losing benefits of residual connection? or do it all before the attention layers?
            out = self.fNNs[i](out) + out
            if hasattr(self, "downsampling_layers") and (
                self.downsample_every_layer or not i % 2
            ):
                j = i
                if not self.downsample_every_layer:
                    j //= 2
                out = self.downsampling_layers[j](out)
            if i != self.n_blocks - 1:
                out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                    -1, -3
                )  # layernorm
        if intermediate:
            return out, torch.stack(output, -1)
        # if debug:
        #     return out, final_pattern_count
        return out


class EquivariantStack(nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_heads=4,
        n_layers=6,
        axial=False,
        invariant=True,
        activation="elu",
        attention_heads=4,
        **kwargs,
    ):
        super().__init__()
        self.n_blocks = n_layers
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
        self.site_equivariant_layers = nn.ModuleList()
        if axial:
            self.taxa_equivariant_layers = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.fNNs = nn.ModuleList()
        for i in range(n_layers):
            self.site_equivariant_layers.append(
                EquivariantLayer(heads=n_heads, invariant=False)
            )
            self.layernorms.append(nn.LayerNorm(hidden_channels))

            # do we really need fNNs here?
            self.fNNs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                    activation_layer(),
                )
            )
            if axial:
                self.taxa_equivariant_layers.append(
                    nn.Sequential(
                        PermutationEquivariantLayer(heads=n_heads), activation_layer()
                    )
                )

        if invariant:
            self.site_equivariant_layers.append(
                EquivariantLayer(heads=n_heads, invariant=True)
            )

    def forward(self, out):
        """input must be shape batches x channels x n_pairs x 2*n_sites"""
        *_, L = out.shape
        d = L // 2
        ix = torch.cat(
            (torch.arange(d, L, device=out.device), torch.arange(d, device=out.device))
        )  # reuse index across layers

        for i in range(self.n_blocks):
            residual_out = out / self.n_blocks
            out = self.activation(
                self.site_equivariant_layers[i](out, ix)
            )  # + out/ self.n_blocks  # Sequential can only take 1 arg
            if hasattr(self, "taxa_equivariant_layers"):
                out = self.taxa_equivariant_layers[i](out)  # + out/ self.n_blocks
            out = self.fNNs[i](out) + residual_out  # + out/ self.n_blocks
            out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                -1, -3
            )  # layernorm
        if self.invariant:
            out = self.site_equivariant_layers[-1](out, ix)
        return out


class EquivariantCompressedStack(nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_heads=4,
        axial=False,
        latent_dims: List = [32],
        cross_attention=False,
        activation="elu",
        **kwargs,
    ):
        super().__init__()
        latent_dims.append(1)

        if activation == "elu":
            activation_layer = nn.ELU
            self.activation = F.elu
        elif activation == "relu":
            activation_layer = nn.ReLU
            self.activation = F.relu
        elif activation == "gelu":
            activation_layer = nn.GELU
            self.activation = F.gelu

        self.n_blocks = len(latent_dims)
        if axial:
            self.taxa_equivariant_layers = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.fNNs = nn.ModuleList()
        self.compressed_attention_layers = nn.ModuleList()
        self.site_equivariant_layers = nn.ModuleList()

        for i in range(self.n_blocks):
            # use positional encoding only in the first layer of the parent
            # TODO: add option for 1D CNN instead of attention
            self.compressed_attention_layers.append(
                nn.Sequential(
                    CompressedMultiAttention(
                        hidden_channels,
                        n_heads=n_heads,
                        latent_dim=latent_dims[0],
                    ),
                    activation_layer(),
                )
            )
            self.site_equivariant_layers.append(
                EquivariantLayer(heads=n_heads, invariant=False)
            )
            self.layernorms.append(nn.LayerNorm(hidden_channels))

            self.fNNs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                    activation_layer(),
                )
            )
            if axial:
                self.taxa_equivariant_layers.append(
                    nn.Sequential(
                        PermutationEquivariantLayer(heads=n_heads), activation_layer()
                    )
                )

        self.site_equivariant_layers.append(
            EquivariantLayer(heads=n_heads, invariant=True)
        )

        # self.final_layer = nn.Linear(
        #     in_features=latent_dims[-1], out_features=1)
        if cross_attention:
            pass
            # TODO: use activations from entire stack

    def forward(self, out):
        """input must be shape batches x channels x n_pairs x 2*n_sites"""
        # reuse index across layers
        # scale residuals by number of blocks a la https://proceedings.neurips.cc/paper/2018/hash/d81f9c1be2e08964bf9f24b15f0e4900-Abstract.html

        for i in range(self.n_blocks):
            residual_out = out / self.n_blocks

            out = self.compressed_attention_layers[i](
                out
            )  # new dim, so can't do residual

            *_, L = out.shape
            d = L // 2
            ix = torch.cat(
                (
                    torch.arange(d, L, device=out.device),
                    torch.arange(d, device=out.device),
                )
            )

            out = self.activation(
                self.site_equivariant_layers[i](out, ix)
            )  # + out / self.n_blocks

            if hasattr(self, "taxa_equivariant_layers"):
                out = self.taxa_equivariant_layers[i](out)  # + out/ self.n_blocks

            out = self.fNNs[i](out) + residual_out  # out / self.n_blocks

            out = self.layernorms[i](out.transpose(-1, -3)).transpose(-1, -3)

        out = self.site_equivariant_layers[-1](out, ix)

        # out = self.final_layer(out)
        return out


class EquivariantNet(nn.Module):
    """Simplified equivariant net"""

    def __init__(
        self,
        params,
        dropout=0.0,
        char_embedding_dim: int = 22,
        format="distance",
        preprocessor=None,
        long_embedding=False,
        positional=False,
        transfer=False,
        activation="elu",
        softplus=True,
        **kwargs,
    ):
        """4-param equivariant distance network
        Args:
            everything in the argument params will be passed to the equivariant stack, only the 'hidden channels' parameter is accessed here.
            dropout (float, optional): dropout. Defaults to 0.0.
            hidden_channels (int, optional): per-site features. Defaults to 64.
            char_embedding_dim (int, optional): initial character embedding. Defaults to 22.
            attention_heads (int, optional): num attn heads. Defaults to 4.
            n_attention_layers (int, optional): num attn layers. Defaults to 6.
            positional (bool, optional): whether to add positional encoding to first layer. Defaults to False.
            kernel_size (int, optional): size of kernel in coarsening; only used if stride>1. Defaults to 5.
            stride (int, optional): If stride > 1, coarsens graph by adding CNN of this stride after every other attn layer. Defaults to 2.
            if transfer is true, will not use embedding layer and *will not* convert seqs to pairs.
        """
        # TODO: refactor everything so we maintain 2 separate matrices for seq1 and seq2; forward applies same transformations to both except for the site_equivariant layers
        super().__init__()
        hidden_channels = params["hidden_channels"]
        # heads = params['heads']
        self.dropout = dropout
        self.as_distances = format == "distance"
        self.preprocess = preprocessor is not None  # use frozen preprocessor
        self.transfer = transfer
        # if attention_params.get('positional', False) and not attention_params.get('local', False) and attention_params.get('field_size_row', 0) == 0:
        if "latent_dims" in params:
            positional = True  # since seq positions are lost when we compress
            self.equivariant = EquivariantCompressedStack(
                activation=activation, **params
            )
        else:
            self.equivariant = EquivariantStack(activation=activation, **params)

        if positional:
            self.pos_encoding = PositionalEncoding(hidden_channels, max_len=5000)

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

        if self.preprocess:
            self.embed = preprocessor.embed
            self.block_1_1 = preprocessor.block_1_1
            self.norm = preprocessor.norm
            for module in (self.embed, self.block_1_1, self.norm):
                for param in module.parameters():
                    param.requires_grad = False
            print("loaded preprocessor")
        elif not transfer:

            self.embed = nn.Embedding(
                num_embeddings=22, embedding_dim=char_embedding_dim
            )
            if long_embedding:

                self.block_1_1 = nn.Sequential(
                    # nn.Dropout(dropout),
                    nn.Conv2d(
                        in_channels=char_embedding_dim,
                        out_channels=hidden_channels * 2,
                        kernel_size=1,
                        stride=1,
                    ),
                    # activation_layer(),
                    # nn.Conv2d(
                    #     in_channels=hidden_channels*4,
                    #     out_channels=hidden_channels * 2 ,
                    #     kernel_size=1,
                    #     stride=1,
                    # ),
                    activation_layer(),
                    nn.Conv2d(
                        in_channels=hidden_channels * 2,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                    activation_layer(),
                )
            else:

                self.block_1_1 = nn.Sequential(
                    nn.Conv2d(
                        in_channels=char_embedding_dim,
                        out_channels=hidden_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                    nn.Dropout(dropout),
                    activation_layer(),
                )
            # self.norm = nn.LayerNorm(hidden_channels)

        # TODO: make pw_layer which up-samples dim of invariant layer output
        pw_layers = [
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_channels, out_features=2 * hidden_channels),
            activation_layer(),
            nn.Linear(in_features=2 * hidden_channels, out_features=hidden_channels),
            activation_layer(),
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels // 2),
            activation_layer(),
            nn.Linear(in_features=hidden_channels // 2, out_features=1),  # //heads,
            nn.ReLU(),  # enforce positive output
        ]
        # if softplus:
        #     pw_layers.append(nn.Softplus())
        self.pwFNN = nn.Sequential(*pw_layers)

        self.S = 0.0
        self.n = 0.0

    @lru_cache
    def seq2pair(self, nb_seq: int, device: torch.device | str | None = None):
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

    @torch.compiler.disable(recursive=True)
    def compute_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        ntaxa = x.size(1)
        out = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )
        out = self.block_1_1(out)

        if hasattr(self, "pos_encoding"):
            out = self.pos_encoding(out)

        ix = self.seq2pair(ntaxa, out.device)
        # out = torch.index_select(
        #     out, dim=-1, index=ix
        # )  # fix...
        out = torch.cat([out[..., ix[0], :], out[..., ix[1], :]], -1)

        # from here on the tensor has shape (batch_size, features, nb_pairs, 2*seq_len)

        return out

    def forward(self, x: torch.Tensor):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        if x.dim() < 3:
            x = x.unsqueeze(0)
        if not hasattr(self, "transfer") or not self.transfer:
            x = self.compute_embeddings(x)
        x = self.activation(x)
        x = self.equivariant(x)  # shape: batch_size x channels x nb_pairs
        x = self.activation(x)
        # after this last convolution we have (batch_size,nb_pairs,1)
        x = self.WN(x.transpose(-1, -2))
        # after this last op we have (batch_size,nb_pairs)
        return x.squeeze()


class AttentionNet(nn.Module):
    """Original Phyloformer Network"""

    def __init__(
        self,
        attention_params,
        dropout=0.0,
        char_embedding_dim: int = 22,
        format="distance",
        pw_dim=None,
        preprocessor=None,
        positional=False,
        transfer=False,
        softplus=True,
        mean_first=False,
    ):
        """axial attention with optional coarsening
        Args:
            dropout (float, optional): dropout. Defaults to 0.0.
            hidden_channels (int, optional): per-site features. Defaults to 64.
            char_embedding_dim (int, optional): initial character embedding. Defaults to 22.
            attention_heads (int, optional): num attn heads. Defaults to 4.
            n_attention_layers (int, optional): num attn layers. Defaults to 6.
            positional (bool, optional): whether to add positional encoding to first layer. Defaults to False.
            kernel_size (int, optional): size of kernel in coarsening; only used if stride>1. Defaults to 5.
            stride (int, optional): If stride > 1, coarsens graph by adding CNN of this stride after every other attn layer. Defaults to 2.
            if transfer is true, will not use embedding layer and *will not* convert seqs to pairs.
        """
        super().__init__()
        hidden_channels = attention_params["hidden_channels"]
        self.dropout = dropout
        self.as_distances = format == "distance"
        self.preprocess = preprocessor is not None  # use frozen preprocessor
        self.transfer = transfer
        # if attention_params.get('positional', False) and not attention_params.get('local', False) and attention_params.get('field_size_row', 0) == 0:
        if positional:
            self.pos_encoding = PositionalEncoding(hidden_channels, max_len=10000)
        if self.preprocess:
            self.embed = preprocessor.embed
            self.block_1_1 = preprocessor.block_1_1
            self.norm = preprocessor.norm
            for module in (self.embed, self.block_1_1, self.norm):
                for param in module.parameters():
                    param.requires_grad = False
            print("loaded preprocessor")
        elif not transfer:

            self.embed = nn.Embedding(
                num_embeddings=22, embedding_dim=char_embedding_dim
            )

            self.block_1_1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=char_embedding_dim,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=1,
                ),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
            self.norm = nn.LayerNorm(hidden_channels)

        self.attentions = AttentionStack(**attention_params)

        self.mean_first = mean_first
        if mean_first:

            pw_layers = []
            for i in range(4):
                pw_layers.extend(
                    [
                        nn.Dropout(dropout),
                        nn.Conv2d(
                            in_channels=pw_dim or hidden_channels,
                            out_channels=pw_dim or hidden_channels,
                            kernel_size=1,
                            stride=1,
                        ),
                    ]
                )
        else:
            pw_layers = [
                nn.Dropout(dropout),
                nn.Conv2d(
                    in_channels=pw_dim or hidden_channels,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                ),
            ]

        if softplus:
            pw_layers.append(nn.Softplus())
        self.pwFNN = nn.Sequential(*pw_layers)
        # TODO: use vmap
        self.get_pair_distances = torch.vmap(torch.vmap(torch.pdist))

        self.S = 0.0
        self.n = 0.0

    @property
    def baseline(self) -> float:
        """only necessary for q function subclasses"""
        return self.n and self.S / self.n

    def update_baseline(self, r):
        self.S += r.sum()
        self.n += len(r)

    @lru_cache
    def seq2pair(self, nb_seq: int, device=None):
        # no_diagonal: bool = True
        """creates indexer to transform seqs to seq pairs.

        Args:
            nb_seq (_type_): number of seqs

        Returns:
            _type_: _description_
        """
        nb_pairs = int(binom(nb_seq, 2))
        if not self.as_distances:
            nb_pairs += nb_seq

        S = torch.zeros(nb_pairs, nb_seq, device=device)
        k = 0
        for i in range(nb_seq):
            for j in range(i + self.as_distances, nb_seq):
                S[k, i] = 1
                S[k, j] = 1
                k = k + 1

        return S

    def compute_embeddings(self, x):
        ntaxa = x.size(1)
        out = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )
        out = self.block_1_1(out)

        out = torch.matmul(self.seq2pair(ntaxa, out.device), out)

        # from here on the tensor has shape (batch_size,features,nb_pairs,seq_len), all the transpose/permute allow to apply layernorm
        # and attention over the desired dimensions and are then followed by the inverse transposition/permutation of dimensions

        out = self.norm(out.transpose(-1, -3)).transpose(
            -1, -3
        )  # .transpose(-1, -3)  # layernorm
        if hasattr(self, "pos_encoding"):
            out = self.pos_encoding(out)

        return out

    def compute_distances(self, x: torch.Tensor):
        # x = x.squeeze(1)
        # if x.dim() < 3:  # batchsize=1
        #     x = x.unsqueeze(0)
        if self.as_distances:
            # reshape to allow avg over channels and sites
            x = self.get_pair_distances(
                x
            )  # torch.stack([torch.pdist(m.reshape(m.shape[0], -1), 2) for m in x])

        else:  # inner product
            batch_size, N, L = x.shape
            x = torch.einsum("bik,bjk->bij", x, x)
            x = x[:, torch.ones(N, N, dtype=bool).triu(diagonal=0)]
        return x

    def forward(self, x, output=True, debug=False):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        if x.dim() < 3:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        if not hasattr(self, "transfer") or not self.transfer:
            x = self.compute_embeddings(x)
        x = self.attentions(x, debug=debug)
        if debug:
            final_pattern_count = x[0, :, 0, :].unique(dim=-1).shape

        if output:
            if self.mean_first:
                x = x.mean(-1, keepdim=True)
            # after this last convolution we have (batch_size,1,nb_pairs,seq_len) unless mean_first
            x = self.pwFNN(x)
            # averaging over positions and removing the extra dimensions we finally get (batch_size,nb_pairs)
            x = torch.mean(x, dim=-1).view(batch_size, -1)
        if debug:
            return x, final_pattern_count
        return x


class TransferNet(nn.Module):
    """operate on seqs, not pairs"""

    def __init__(self, tail: AttentionNet, head: AttentionNet, **kwargs):
        """axial attention with optional coarsening
        Args:
            dropout (float, optional): dropout. Defaults to 0.0.
            hidden_channels (int, optional): per-site features. Defaults to 64.
            char_embedding_dim (int, optional): initial character embedding. Defaults to 22.
            attention_heads (int, optional): num attn heads. Defaults to 4.
            n_attention_layers (int, optional): num attn layers. Defaults to 6.
            positional (bool, optional): whether to add positional encoding to first layer. Defaults to False.
            kernel_size (int, optional): size of kernel in coarsening; only used if stride>1. Defaults to 5.
            stride (int, optional): If stride > 1, coarsens graph by adding CNN of this stride after every other attn layer. Defaults to 2.
        """
        # NOTE: must preserve order of input args in parent class
        super().__init__(**kwargs)
        self.head = head
        self.head.transfer = True
        self.tail = tail
        # following  “Pretrained Transformers As Universal Computation Engines”, only retrain the layer norms
        for name, param in self.tail.named_parameters():
            param.requires_grad = "norm" in name
        # leads to error: Output 0 of ReshapeAliasBackward0
        # for module in self.tail.modules():
        #     if hasattr(module,'inplace'):module.inplace=True

    # def parameters(self):
    #     """only returns TRAINABLE parameters"""
    #     return self.head.parameters()

    def forward(self, x):
        x = self.tail(x, output=False)
        try:
            # warning: do not use for trainable (e.g. conv) downsampler
            x = self.head.attentions.downsampling_layers[0](x)
        except:
            pass
        x = self.head(x)
        return x


class SeqNet(AttentionNet):
    """operate on seqs, not pairs"""

    def __init__(self, radial=False, **kwargs):
        """axial attention with optional coarsening
        Args:
            dropout (float, optional): dropout. Defaults to 0.0.
            hidden_channels (int, optional): per-site features. Defaults to 64.
            char_embedding_dim (int, optional): initial character embedding. Defaults to 22.
            attention_heads (int, optional): num attn heads. Defaults to 4.
            n_attention_layers (int, optional): num attn layers. Defaults to 6.
            positional (bool, optional): whether to add positional encoding to first layer. Defaults to False.
            kernel_size (int, optional): size of kernel in coarsening; only used if stride>1. Defaults to 5.
            stride (int, optional): If stride > 1, coarsens graph by adding CNN of this stride after every other attn layer. Defaults to 2.
        """
        # NOTE: must preserve order of input args in parent class
        super().__init__(**kwargs)
        # ignored; currently only used by LogDet regularizer to transform dist -> cov
        self.radial = radial

    def compute_embeddings(self, x):
        ntaxa = x.size(1)
        out = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )
        out = self.block_1_1(out)

        # from here on the tensor has shape (batch_size,features,n_taxa,seq_len), all the transpose/permute allow to apply layernorm
        # and attention over the desired dimensions and are then followed by the inverse transposition/permutation of dimensions

        out = self.norm(out.transpose(-1, -3)).transpose(
            -1, -3
        )  # .transpose(-1, -3)  # layernorm
        if hasattr(self, "pos_encoding"):
            out = self.pos_encoding(out)

        return out

    def forward(self, x, debug=False):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        B, *_ = x.shape
        if x.dim() < 3:
            x = x.unsqueeze(0)
        if not self.transfer:
            x = self.compute_embeddings(x)
        x = self.attentions(x)

        if debug:
            final_pattern_count = x[0, :, 0, :].unique(dim=-1).shape

        if self.mean_first:
            x = x.mean(-1, keepdim=True)  # average over positions

        # after this last convolution we have (batch_size,1,n_taxa,seq_len) unless mean_first, then (batch_size,h_dim,n_taxa,1)
        x = self.pwFNN(x)

        # averaging over positions and removing the extra dimensions we finally get (batch_size,nb_pairs)
        if self.mean_first:
            x = x.transpose(-1, -3)
        x = self.compute_distances(x)
        x = x.reshape(B, -1)

        if debug:
            return x, final_pattern_count
        return x


class FreqNet(SeqNet):
    """corrected distance + AA frequency feature; each site knows freqs of"""

    def __init__(self, **kwargs):
        h = kwargs["attention_params"]["hidden_channels"]
        super().__init__(**kwargs, pw_dim=1 + h)
        self.freq_net = build_fc_network(
            22, 22, [22], batch_norm=False, layer_norm=True, nonlinearity=nn.GELU
        )

    def forward(self, x: torch.Tensor, output=True):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)

        if not x.dim() == 3:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        # per-seq frequencies
        x = x.long()
        freqs = F.one_hot(x, num_classes=22).float().mean(-2)

        freqs = self.freq_net(freqs)
        # each site knows the global frequency of all AA's
        freqs = torch.gather(freqs, dim=2, index=x)
        x = self.compute_embeddings(x)
        x = self.attentions(x)
        x = torch.cat([x, freqs.unsqueeze(1)], 1)
        if output:
            # after this last convolution we have (batch_size,1,nb_pairs,seq_len)
            x = self.pwFNN(x)
            # averaging over positions and removing the extra dimensions we finally get (batch_size,nb_pairs)
            # x = torch.mean(x, dim=-1).view(batch_size, -1)
            x = self.compute_distances(x)
        return x


class FreqNet2(SeqNet):
    """corrected distance + AA frequency feature - each site knows freqs of _all_ AA's"""

    def __init__(self, **kwargs):
        h = kwargs["attention_params"]["hidden_channels"]
        super().__init__(**kwargs, pw_dim=32 + h)
        self.freq_net = build_fc_network(
            22, 32, [32], batch_norm=False, layer_norm=True, nonlinearity=nn.GELU
        )

    def forward(self, x: torch.Tensor, output=True):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)

        if not x.dim() == 3:
            x = x.unsqueeze(0)
        B, N, L = x.shape

        # per-seq frequencies
        x = x.long()
        freqs = F.one_hot(x, num_classes=22).float().mean(-2)

        freqs = self.freq_net(freqs)
        # each site knows the global frequency of all AA's
        x = self.compute_embeddings(x)
        x = self.attentions(x)
        x = torch.cat([x, freqs.transpose(1, 2).unsqueeze(-1).expand(B, -1, N, L)], 1)
        # after this last convolution we have (batch_size,1,n_taxa,seq_len)
        if output:
            x = self.pwFNN(x)
            # averaging over positions and removing the extra dimensions we finally get (batch_size,nb_pairs)
            # x = torch.mean(x, dim=-1).view(batch_size, -1)
            x = self.compute_distances(x)
        return x


class SeqPairNet(SeqNet):
    """operate on seqs THEN pairs.
    if attention_params_2 is None, this is equivalent to a SeqNet"""

    def __init__(
        self,
        attention_params: dict,
        attention_params_2: dict = None,
        radial=False,
        dropout=0.0,
        softplus=True,
        pw_dim=None,
        **kwargs,
    ):
        """axial attention with optional coarsening
        Args:
            dropout (float, optional): dropout. Defaults to 0.0.
            hidden_channels (int, optional): per-site features. Defaults to 64.
            char_embedding_dim (int, optional): initial character embedding. Defaults to 22.
            attention_heads (int, optional): num attn heads. Defaults to 4.
            n_attention_layers (int, optional): num attn layers. Defaults to 6.
            positional (bool, optional): whether to add positional encoding to first layer. Defaults to False.
            kernel_size (int, optional): size of kernel in coarsening; only used if stride>1. Defaults to 5.
            stride (int, optional): If stride > 1, coarsens graph by adding CNN of this stride after every other attn layer. Defaults to 2.
            preprocessor : if provided, will copy embed and 1st attention stack from this network.
        """
        # NOTE: must preserve order of input args in parent class
        super().__init__(
            dropout=dropout,
            softplus=softplus,
            attention_params=attention_params,
            **kwargs,
        )
        # ignored; currently only used by LogDet regularizer to transform dist -> cov
        self.radial = radial

        preprocessor = kwargs.get("preprocessor", None)

        if preprocessor and hasattr(preprocessor, "attentions"):
            self.attentions = preprocessor.attentions
            for param in self.attentions.parameters():
                param.requires_grad = False

        # del preprocessor
        hidden_channels_1 = attention_params["hidden_channels"]
        if attention_params_2 is not None:
            hidden_channels_2 = attention_params_2["hidden_channels"]

            self.attentions_2 = AttentionStack(**attention_params_2)

            pw_layers = [
                nn.Dropout(dropout),
                nn.Conv2d(
                    in_channels=pw_dim or hidden_channels_2,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                ),
            ]
            if softplus:
                pw_layers.append(nn.Softplus())
            self.pwFNN = nn.Sequential(*pw_layers)

            self.seq_FNN = nn.Sequential(
                nn.Dropout(dropout),
                nn.Conv2d(
                    in_channels=hidden_channels_1,
                    out_channels=hidden_channels_2,
                    kernel_size=1,
                    stride=1,
                ),
                nn.GELU(),
            )
            self.norm_2 = nn.LayerNorm(hidden_channels_2)

    def compute_embeddings(self, x):
        ntaxa = x.size(1)
        out = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )
        out = self.block_1_1(out)

        # from here on the tensor has shape (batch_size,features,n_taxa,seq_len), all the transpose/permute allow to apply layernorm
        # and attention over the desired dimensions and are then followed by the inverse transposition/permutation of dimensions

        out = self.norm(out.transpose(-1, -3)).transpose(
            -1, -3
        )  # .transpose(-1, -3)  # layernorm
        if hasattr(self, "pos_encoding"):
            out = self.pos_encoding(out)

        return out

    def compute_second_attentions(self, x, debug=False):
        *_, ntaxa, _ = x.shape
        x = self.seq_FNN(x)

        x = self.norm_2(x.transpose(-1, -3)).transpose(-1, -3)

        x = torch.matmul(self.seq2pair(ntaxa, x.device), x)
        x = self.attentions_2(x, debug=debug)
        # try:
        #     x = self.attentions_2(x)
        # except:
        #     raise
        # torch.cuda.empty_cache()
        # x = self.attentions_2(x)
        return x

    def forward(self, x, intermediate=False, output=True, debug=False):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        if x.dim() < 3:
            x = x.unsqueeze(0)
        if not self.transfer:
            x = self.compute_embeddings(x)
        # if debug:
        #     print("embed", x[0, :, 0, :].unique(dim=-1).shape)
        # if intermediate:
        #     x, out = self.compute_attentions(x, intermediate, debug)
        #     x = self.pwFNN(x)
        #     return torch.cat([out, self.compute_distances(x).unsqueeze(-1)], -1)

        x = self.attentions(x, debug=debug)
        batch_size = x.shape[0]

        if hasattr(self, "attentions_2"):
            x = self.compute_second_attentions(x, debug)
            if debug:
                final_pattern_count = x[0, :, 0, :].unique(dim=-1).shape
            # after this last convolution we have (batch_size,1,n_taxa,seq_len)
            if self.mean_first:
                x = x.mean(-1, keepdim=True)  # average over positions
            if output:
                x = self.pwFNN(x)
                x = torch.mean(x, dim=-1).view(batch_size, -1)
            if debug:
                return x, final_pattern_count
            return x

        if output:
            # after this last convolution we have (batch_size,1,n_taxa,seq_len)

            # TODO: would it be better to average over seq_len?
            x = self.pwFNN(x)
            if debug:
                print("pwFNN", x[0, :, 0, :].unique(dim=-1).shape)
            x = self.compute_distances(x)

        return x


class FreqPairNet(SeqPairNet):
    """corrected distance + AA frequency feature - each site knows freqs of _all_ AA's"""

    def __init__(self, **kwargs):
        h = kwargs["attention_params"]["hidden_channels"]
        super().__init__(**kwargs, pw_dim=22 + h)
        self.freq_net = build_fc_network(
            22, 22, [32], batch_norm=False, layer_norm=True, nonlinearity=nn.GELU
        )

    def forward(self, x: torch.Tensor, output=True):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)

        if not x.dim() == 3:
            x = x.unsqueeze(0)
        B, N, L = x.shape

        # per-seq frequencies
        x = x.long()
        freqs = F.one_hot(x, num_classes=22).float().mean(-2)

        freqs = self.freq_net(freqs) + freqs
        freqs = self.seq2pair(N, freqs.device) @ freqs  # B,N,D

        # each site knows the global frequency of all AA's
        x = self.compute_embeddings(x)
        x = self.attentions(x)
        if hasattr(self, "attentions_2"):
            x = self.compute_second_attentions(x)

        # freqs=freqs.unsqueeze(-1).expand(B,-1,N,L)
        x = torch.cat([x, freqs.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, L)], 1)
        # after this last convolution we have (batch_size,1,nb_pairs,seq_len)
        if output:
            x = self.pwFNN(x)
            # averaging over positions and removing the extra dimensions we finally get (batch_size,nb_pairs)
            x = torch.mean(x, dim=-1).view(B, -1)

        return x
        # return self.compute_distances(x)


class LocalNet(AttentionNet):
    """Original Phyloformer Network"""

    def __init__(self, positional=False, **kwargs):
        """axial attention with optional coarsening
        Args:
            dropout (float, optional): dropout. Defaults to 0.0.
            hidden_channels (int, optional): per-site features. Defaults to 64.
            char_embedding_dim (int, optional): initial character embedding. Defaults to 22.
            attention_heads (int, optional): num attn heads. Defaults to 4.
            n_attention_layers (int, optional): num attn layers. Defaults to 6.
            positional (bool, optional): whether to add positional encoding to first layer. Defaults to False.
            kernel_size (int, optional): size of kernel in coarsening; only used if stride>1. Defaults to 5.
            stride (int, optional): If stride > 1, coarsens graph by adding CNN of this stride after every other attn layer. Defaults to 2.
        """
        # NOTE: must preserve order of input args in parent class
        super().__init__(**kwargs)
        n_h, n_attn, f = (
            kwargs["hidden_channels"],
            kwargs["attention_heads"],
            kwargs["field_size_row"],
        )
        self.row_attentions = nn.ModuleList()
        for i in range(self.n_blocks):
            self.row_attentions.append(
                LocalAttentionBlock(n_h, n_attn, pos_embed=positional, field_size=f)
            )


class SDPANet(AttentionNet):
    """Original Phyloformer Network"""

    def __init__(
        self,
        dropout=0.0,
        hidden_channels: int = 64,
        char_embedding_dim: int = 22,
        attention_heads: int = 4,
        n_attention_layers=6,
        positional=False,
        kernel_size=5,
        downsample="every_layer",
        downsample_strategy="conv",
        preprocessor=None,
        format="distance",
        field_size_row=None,
        field_size_col=None,
        row=True,
        with_q=False,
        unfold=False,
        col=True,
        softplus=True,
        stride=1,
    ):
        """axial attention with optional coarsening
        Args:
            dropout (float, optional): dropout. Defaults to 0.0.
            hidden_channels (int, optional): per-site features. Defaults to 64.
            char_embedding_dim (int, optional): initial character embedding. Defaults to 22.
            attention_heads (int, optional): num attn heads. Defaults to 4.
            n_attention_layers (int, optional): num attn layers. Defaults to 6.
            positional (bool, optional): whether to add positional encoding to first layer. Defaults to False.
            kernel_size (int, optional): size of kernel in coarsening; only used if stride>1. Defaults to 5.
            stride (int, optional): If stride > 1, coarsens graph by adding CNN of this stride after every other attn layer. Defaults to 2.
        """
        # NOTE: must preserve order of input args in parent class
        super().__init__(
            dropout,
            hidden_channels,
            char_embedding_dim,
            attention_heads,
            n_attention_layers,
            positional,
            kernel_size,
            downsample,
            downsample_strategy,
            preprocessor,
            format,
            field_size_row,
            field_size_col,
            row,
            with_q,
            unfold,
            col,
            softplus,
            stride,
        )
        fNN_channels = attention_heads * hidden_channels
        if row:
            self.row_attentions = nn.ModuleList()
        if col:
            self.column_attentions = nn.ModuleList()

        for i in range(self.n_blocks):
            if row:
                self.row_attentions.append(
                    AxialAttention(
                        hidden_channels,
                        attention_heads,
                        field_size=field_size_row,
                        with_q=with_q,
                    )
                )
            if col:
                self.column_attentions.append(
                    AxialAttention(
                        hidden_channels,
                        attention_heads,
                        unfold=unfold,
                        field_size=field_size_col,
                        with_q=with_q,
                    )
                )
