from pandas import infer_freq
import torch.nn.functional as F
from einops import rearrange
from pathlib import Path
from typing import Literal
from torch import nn
import torch_geometric.utils as gm_utils
from torch_geometric.nn import (
    GATv2Conv,
    GraphMultisetTransformer,
    SuperGATConv,
    pool,
    global_mean_pool,
)
from torch_geometric.utils import to_undirected
from .models.edge_conv import *
from .models.graph_utils import *

import math
from functools import lru_cache
from torch_geometric.nn import GATv2Conv
import torch
import torch.nn as nn
from scipy.special import binom
from .models.layers import (
    AxialAttention,
    CompressedMultiAttention,
    EquivariantLayer,
    KernelAxialMultiAttention,
    LinearMultiAttention,
    LocalAttentionBlock,
    PermutationEquivariantLayer,
    RelPosEmb,
    rel_pos_emb_1d,
)
from .models.graph_utils import (
    EmbedLayer,
    make_batch_indices,
    make_conv_net,
    make_conv_net_2d,
    make_indices,
)


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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # final shape: 1 x <d_model> x 1 x<max_len>
        pe = pe.t().unsqueeze(1).unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)
        self.parameter = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = x + self.pe[..., : x.shape[-1]] * self.parameter
        return x


class LocalAttentionBlock(nn.Module):
    def __init__(
        self,
        h_dim,
        n_heads,
        dropout=0.0,
        pos_embed=False,
        share_qk=False,
        field_size=None,
    ):
        """axial attention with linear softmax approximation

        Args:
            h_dim (int ): must be divisible by n_heads
            n_heads (_type_): _description_
            dropout (float, optional): _description_. Defaults to 0.0.
            eps (_type_, optional): _description_. Defaults to 1e-6.
            pos_embed (bool, optional): _description_. Defaults to False.
            field_size (_type_, optional): if specified, attentions will be limited to this many neighbors.  Must be ODD.
             Defaults to None.
            with_q (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        d_head, remainder = divmod(h_dim, n_heads)
        if remainder:
            raise ValueError("incompatible `d_model` and `num_heads`")
        self.heads = n_heads
        self.h_dim = h_dim
        self.qkv_net = nn.Linear(h_dim, h_dim * 3)

        self.elu = nn.ELU()

        self.attn_fn = LocalAttention(
            dim=d_head,
            window_size=field_size,
            causal=False,
            use_xpos=pos_embed,
            look_backward=1,
            look_forward=1,
            autopad=True,
            shared_qk=share_qk,
            use_rotary_pos_emb=pos_embed,
        )
        self.proj_net = nn.Linear(h_dim, h_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # shape (col attn) : batch x seqlen x pairs x features
        # row attn: (batch_size,nb_pairs,seq_len,features)
        q, k, v = self.qkv_net(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "... n (h d) -> ... h n d", h=self.heads), (q, k, v)
        )
        # h,d=self.heads,self.h_dim//self.heads
        # q, k, v = (m.unflatten(-1,(h,d)).transpose(-2,-3) for m in (q,k,v))
        # q=q.reshape(*q.shape[:-1],h,d).transpose(-2,-3)
        # k=k.reshape(*k.shape[:-1],h,d).transpose(-2,-3)
        # v=v.reshape(*v.shape[:-1],h,d).transpose(-2,-3)

        out = self.attn_fn(q, k, v)
        # out= out.transpose(-2,-3).flatten(-2)
        out = rearrange(out, "... h n d -> ... n (h d)")
        a = []
        out = self.proj_drop(self.proj_net(out))
        return out, a


class CompressedAttentionNet(nn.Module):
    """Phyloformer Network"""

    def __init__(
        self,
        char_embedding_dim=16,
        hidden_channels=64,
        compressed_length=150,
        conv_layer_sizes=[128, 128],
        kernel_size=16,
        attention_heads=4,
        stride=3,
        n_attention_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.n_blocks = n_attention_layers
        self.row_attentions = nn.ModuleList()
        self.column_attentions = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.fNNs = nn.ModuleList()
        self.dropout = dropout

        self.embed = nn.Embedding(num_embeddings=22, embedding_dim=char_embedding_dim)
        layers_1_1 = [
            make_conv_net_2d(
                [char_embedding_dim] + conv_layer_sizes, kernel_size, stride, dropout
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=conv_layer_sizes[-1],
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.AdaptiveAvgPool2d((None, compressed_length)),
        ]
        self.block_1_1 = nn.Sequential(*layers_1_1)
        self.norm = nn.LayerNorm(hidden_channels)
        self.pwFNN = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=1, kernel_size=1, stride=1
            ),
            nn.Dropout(dropout),
            nn.Softplus(),
        )
        fNN_channels = hidden_channels * attention_heads
        for i in range(self.n_blocks):
            self.row_attentions.append(
                KernelAxialMultiAttention(hidden_channels, attention_heads)
            )
            self.column_attentions.append(
                KernelAxialMultiAttention(hidden_channels, attention_heads)
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

    @torch.compiler.disable
    @lru_cache(maxsize=3)  # disable caching for very large alignments
    def seq2pair(self, nb_seq, device=None):
        """creates indexer to transform seqs to seq pairs.

        Args:
            nb_seq (_type_): _description_

        Returns:
            _type_: _description_
        """
        nb_pairs = int(binom(nb_seq, 2))

        S = torch.zeros(nb_pairs, nb_seq, device=device)
        k = 0
        for i in range(nb_seq):
            for j in range(i + 1, nb_seq):
                S[k, i] = 1
                S[k, j] = 1
                k = k + 1

        return S

    def forward(self, x, output=True):
        # TODO: handle batches with different numbers of taxa per alignment; i.e. with GNN and a batch index
        # TODO: extend this so we can make a Q function that takes an (alignment, distance_matrix) pair; concat to new col in out after block_1_1
        # x hash shape batches x ntaxa x seqlength
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        if x.dim() == 2:
            x = x[None, ...]
        batch_size, ntaxa, seq_length = x.shape

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

        out = self.norm(out.transpose(-1, -3)).transpose(-1, -3)  # layernorm

        out = self.compute_attentions(out)
        if output:
            # after this last convolution we have (batch_size,1,nb_pairs,seq_len)
            out = self.pwFNN(out)
            # averaging over positions and removing the extra dimensions we finally get (batch_size,nb_pairs)
            out = torch.squeeze(torch.mean(out, dim=-1))
        return out

    def compute_attentions(self, out):
        for i in range(self.n_blocks):
            # AXIAL ATTENTIONS BLOCK
            # ----------------------
            # ROW ATTENTION
            att, a = self.row_attentions[i](out.permute(0, 2, 3, 1))
            # row attention+residual connection
            out = att.permute(0, 3, 1, 2) + out
            out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                -1, -3
            )  # layernorm

            # COLUMN ATTENTION
            att, a = self.column_attentions[i](out.permute(0, 3, 2, 1))

            # column attention+residual connection
            out = att.permute(0, 3, 2, 1) + out
            out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                -1, -3
            )  # layernorm

            # FEEDFORWARD
            out = self.fNNs[i](out) + out
            if i != self.n_blocks - 1:
                out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                    -1, -3
                )  # layernorm
        return out


class QFunctionGAT(nn.Module):
    def __init__(
        self,
        char_embedding_dim=16,
        hidden_channels=16,
        compressed_length=50,
        conv_layer_sizes=[64, 32],
        n_attention_layers=2,
        kernel_size=7,
        attention_heads=2,
        stride=2,
        base=3,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=22, embedding_dim=char_embedding_dim)
        self.block_1_1 = nn.Sequential(
            make_conv_net_2d(
                [char_embedding_dim] + conv_layer_sizes, kernel_size, stride, dropout
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=conv_layer_sizes[-1],
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.AdaptiveAvgPool2d((None, compressed_length)),
            #  nn.Flatten()
        )
        self.norm = nn.LayerNorm(hidden_channels)
        flattened_size = hidden_channels * compressed_length
        # TODO: use pyg PositionalEncoding layer

        # TODO use EdgeConv we wrote rather than the GATv2Conv extension to handle n-dim tensors
        self.gats = nn.ModuleList(
            [
                GATv2Conv(
                    flattened_size,
                    flattened_size // base,
                    heads=attention_heads,
                    edge_dim=1,
                    fill_value=0,
                    bias=True,
                    share_weights=True,  # undirected graph
                    dropout=dropout,
                )
            ]
        )
        # self.gat_layer_norms = nn.ModuleList()

        for i in range(1, n_attention_layers):
            input_layer_size = attention_heads * (flattened_size // base**i)
            output_layer_size = flattened_size // base ** (i + 1)
            if output_layer_size == 0:
                raise ValueError(
                    "base, n_attention_layers must be specified s.t. final gat layer is not empty"
                )

            self.gats.append(
                GATv2Conv(
                    input_layer_size,
                    output_layer_size,
                    heads=attention_heads,
                    edge_dim=1,
                    fill_value=0,
                    bias=True,
                    share_weights=True,  # undirected graph
                    dropout=dropout,
                )
            )
            # self.gat_layer_norms.append(nn.LayerNorm(output_layer_size))
            # TODO: try using edge attention weights as output OR add edge GCN
        input_layer_size = attention_heads * output_layer_size
        self.fc = nn.Sequential(
            # nn.LeakyReLU(),
            nn.LayerNorm(input_layer_size),
            nn.Linear(
                in_features=input_layer_size, out_features=input_layer_size, bias=False
            ),
            nn.LeakyReLU(),
            nn.Linear(in_features=input_layer_size, out_features=1, bias=True),
            # nn.ReLU()  # TODO: remove
        )
        #               flattened_size//2**(n_attention_layers+1)),
        #     # nn.Conv1d(in_channels=hidden_channels,
        #     #           out_channels=1, kernel_size=1, stride=1),
        #     nn.Linear(flattened_size//2**(n_attention_layers+1), 1),
        # )

    def forward(self, x: torch.Tensor, dmat: torch.Tensor):
        # TODO: extend this so we can make a Q function that takes an (alignment, distance_matrix) pair; concat to new col in out after block_1_1
        # x hash shape (batches x) ntaxa x seqlength
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        # dmat has shape (batches x) ntaxa x ntaxa

        batch_size, n_seqs, seq_length = x.shape

        out = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )
        out = self.block_1_1(out)
        out = (
            self.norm(out.transpose(-1, -3))
            .transpose(-1, -3)
            .reshape((batch_size * n_seqs, -1))
        )  # layernorm

        indices = make_indices(batch_size, n_seqs, directed=False)
        dmat = dmat.repeat((1, 2))  # undirected network
        dmat = dmat.ravel()
        # edge_attr = torch.hstack([dmat, dmat])
        # out = out.reshape((n_seqs*batch_size, -1))
        # distances = dmat[indices[0], indices[1]]
        # distances = dmat[:, indices[0], indices[1]]
        for gat in self.gats:
            out = gat(out, indices.to(out.device), edge_attr=dmat.to(out.device))
            # out = norm(out)
        # out = out.reshape((n_seqs, batch_size, -1, seq_length))
        out = self.fc(out).reshape((batch_size, n_seqs))
        # out = -out.mean(-1)  # RELU followed by -, since log-likelihood <=0
        # TODO: replace with:
        out = out.mean(-1) - 7000  # scaling # .relu()

        return out


class GAT(nn.Module):
    def __init__(
        self,
        char_embedding_dim=16,
        hidden_channels=16,
        compressed_length=50,
        conv_layer_sizes=[64, 32],
        n_attention_layers=2,
        kernel_size=7,
        attention_heads=2,
        stride=2,
        base=3,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=22, embedding_dim=char_embedding_dim)
        self.block_1_1 = nn.Sequential(
            make_conv_net_2d(
                [char_embedding_dim] + conv_layer_sizes, kernel_size, stride, dropout
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=conv_layer_sizes[-1],
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.AdaptiveAvgPool2d((None, compressed_length)),
            #  nn.Flatten()
        )
        self.norm = nn.LayerNorm(hidden_channels)
        flattened_size = hidden_channels * compressed_length
        # TODO: use pyg PositionalEncoding layer

        # TODO use EdgeConv we wrote rather than the GATv2Conv extension to handle n-dim tensors
        self.gats = nn.ModuleList(
            [
                GATv2Conv(
                    flattened_size,
                    flattened_size // base,
                    heads=attention_heads,
                    bias=True,
                    share_weights=True,  # undirected graph
                    dropout=dropout,
                )
            ]
        )
        # self.gat_layer_norms = nn.ModuleList()

        for i in range(1, n_attention_layers):

            input_layer_size = attention_heads * (flattened_size // base**i)
            output_layer_size = flattened_size // base ** (i + 1)
            if output_layer_size == 0:
                raise ValueError(
                    "base, n_attention_layers must be specified s.t. final gat layer is not empty"
                )

            self.gats.append(
                GATv2Conv(
                    input_layer_size,
                    output_layer_size,
                    heads=attention_heads,
                    bias=True,
                    share_weights=True,  # undirected graph
                    dropout=dropout,
                )
            )
            # self.gat_layer_norms.append(nn.LayerNorm(output_layer_size))

    def forward(self, x: torch.Tensor):
        # TODO: extend this so we can make a Q function that takes an (alignment, distance_matrix) pair; concat to new col in out after block_1_1
        # x hash shape (batches x) ntaxa x seqlength
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        # dmat has shape (batches x) ntaxa x ntaxa

        batch_size, n_seqs, seq_length = x.shape

        out = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )
        out = self.block_1_1(out)
        out = (
            self.norm(out.transpose(-1, -3))
            .transpose(-1, -3)
            .reshape((batch_size * n_seqs, -1))
        )  # layernorm

        indices = make_indices(batch_size, n_seqs, directed=False)

        for gat in self.gats:
            out = F.leaky_relu(out)
            out = gat(out, indices.to(out.device))
            # out = norm(out)
        # out = out.reshape((n_seqs, batch_size, -1, seq_length))
        out = out.reshape((batch_size, n_seqs, -1))
        out = torch.stack([torch.pdist(x, 2) for x in out])
        # out = -out.mean(-1)  # RELU followed by -, since log-likelihood <=0
        # TODO: replace with:
        # out = out.mean(-1)  # .relu()

        return out


class Preprocessor(nn.Module):
    """always on CPU"""

    def __init__(self, network: Union[nn.Module, Path], device) -> None:
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
        compressed_row_sizes: list = None,
        field_size_col: int = None,
        field_size_row: int = None,
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

        if positional:
            d_head, remainder = divmod(hidden_channels, attention_heads)
            self.pos_embed = RelPosEmb(dim_head=d_head, heads=attention_heads)

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
                        # pos_embed=positional,
                        sdpa=sdpa,
                        n_output=compressed_row_sizes[i],
                    )
                elif local:
                    row_block = LocalAttentionBlock(
                        hidden_channels,
                        attention_heads,
                        share_qk=share_qk,
                        # pos_embed=positional,
                        field_size=field_size_row,
                    )
                else:
                    if field_size_row is not None:
                        raise NotImplementedError("field_size_row requires local==True")
                    row_block = KernelAxialMultiAttention(
                        hidden_channels,
                        attention_heads,
                        # pos_embed=positional,
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

    def forward(self, out, col_mask=None, intermediate=False):
        if intermediate:
            output = []
        if hasattr(self, "pos_embed"):
            # add positional encoding to the first layer
            out = self.pos_embed(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
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

            # COLUMN ATTENTION
            if hasattr(self, "column_attentions"):
                # (batch_size,features,nb_pairs,seq_len) -> (batch_size,seq_len,nb_pairs,features)
                att, _ = self.column_attentions[i](out.permute(0, 3, 2, 1), col_mask)

                # column attention+residual connection
                out = att.permute(0, 3, 2, 1) + out
                out = self.layernorms[i](out.transpose(-1, -3)).transpose(
                    -1, -3
                )  # layernorm

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
        return out

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
        x = x.squeeze()
        if x.dim() < 3:  # batchsize=1
            x = x.unsqueeze(0)
        if self.as_distances:
            x = torch.stack([torch.pdist(m, 2) for m in x])

        else:  # inner product
            batch_size, N, L = x.shape
            x = torch.einsum("bik,bjk->bij", x, x)
            x = x[:, torch.ones(N, N, dtype=bool).triu(diagonal=0)]
        return x

    def forward(self, x, output=True):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        if x.dim() < 3:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        if not hasattr(self, "transfer") or not self.transfer:
            x = self.compute_embeddings(x)
        x = self.attentions(x)
        if output:
            # after this last convolution we have (batch_size,1,nb_pairs,seq_len)
            x = self.pwFNN(x)
            # averaging over positions and removing the extra dimensions we finally get (batch_size,nb_pairs)
            x = torch.mean(x, dim=-1).view(batch_size, -1)
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

    def forward(self, x, intermediate=False):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        if x.dim() < 3:
            x = x.unsqueeze(0)
        if not self.transfer:
            x = self.compute_embeddings(x)
        if intermediate:
            x, out = self.compute_attentions(x, intermediate)
            x = self.pwFNN(x)
            return torch.cat([out, self.compute_distances(x).unsqueeze(-1)], -1)

        x = self.attentions(x)

        # after this last convolution we have (batch_size,1,n_taxa,seq_len)
        x = self.pwFNN(x)
        x = self.compute_distances(x)
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

    def compute_second_attentions(self, x):
        *_, ntaxa, _ = x.shape
        x = self.seq_FNN(x)

        x = self.norm_2(x.transpose(-1, -3)).transpose(-1, -3)

        x = torch.matmul(self.seq2pair(ntaxa, x.device), x)
        x = self.attentions_2(x)
        # try:
        #     x = self.attentions_2(x)
        # except:
        #     raise
        # torch.cuda.empty_cache()
        # x = self.attentions_2(x)
        return x

    def forward(self, x, intermediate=False, output=True):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        if x.dim() < 3:
            x = x.unsqueeze(0)
        if not self.transfer:
            x = self.compute_embeddings(x)
        if intermediate:
            x, out = self.compute_attentions(x, intermediate)
            x = self.pwFNN(x)
            return torch.cat([out, self.compute_distances(x).unsqueeze(-1)], -1)

        x = self.attentions(x)
        batch_size = x.shape[0]

        if hasattr(self, "attentions_2"):
            x = self.compute_second_attentions(x)
            # after this last convolution we have (batch_size,1,n_taxa,seq_len)
            if output:
                x = self.pwFNN(x)
                x = torch.mean(x, dim=-1).view(batch_size, -1)
            return x
        if output:
            # after this last convolution we have (batch_size,1,n_taxa,seq_len)
            x = self.pwFNN(x)
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


class AxialGAT(SeqNet):
    def __init__(
        self,
        compressed_length=200,
        concat=True,
        format=Literal["distance", "covariance"],
        **kwargs,
    ):
        """interleave GAT and column attention layers.
        stride must evenly divide hidden_channels*compressed_length since we will do a lot of reshaping

        Args:
            compressed_length (int, optional): compress to (hidden_channels x) compressed_length with adaptive pooling before passing to GAT. Defaults to 200.
            base (int, optional): decrease size by this amount after each GAT. Defaults to 1.
        """
        super().__init__(**kwargs)
        # TODO: like attentionnet, but use gatconv instead of row attention layers
        self.format = format
        # TODO: write new attention layer that looks at each s
        # must reshape to (...,seq_length,hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((compressed_length, None))
        #  nn.Flatten()
        # del self.layernorms  # don't need this since we're flattening all channels
        hidden_channels = kwargs["hidden_channels"]
        # TODO: use pyg PositionalEncoding layer
        attention_heads = kwargs["attention_heads"]
        dropout = kwargs.get("dropout", 0)
        # TODO use EdgeConv we wrote rather than the GATv2Conv extension to handle n-dim tensors
        input_layer_size = hidden_channels * compressed_length
        output_layer_size = input_layer_size
        if concat:
            output_layer_size //= attention_heads

        self.gats = nn.ModuleList()
        for i in range(kwargs["n_attention_layers"]):

            self.gats.append(
                GATv2Conv(
                    input_layer_size,
                    output_layer_size,
                    heads=attention_heads,
                    bias=True,
                    concat=concat,
                    add_self_loops=True,
                    share_weights=True,  # undirected graph
                    dropout=dropout,
                )
            )
            if self.attentions.stride > 1:
                input_layer_size = (
                    math.floor(
                        (
                            output_layer_size // hidden_channels
                            - self.attentions.kernel_size
                        )
                        / self.attentions.stride
                        + 1
                    )
                    * hidden_channels
                )
                output_layer_size = input_layer_size
                if concat:
                    output_layer_size //= attention_heads

            if output_layer_size == 0:
                raise ValueError(
                    "base, n_attention_layers must be specified s.t. final gat layer is not empty"
                )

            # TODO: decide whether to flatten row- or column-major, and how to downsample appropriately over seq_length NOT channels
            # self.gat_layer_norms.append(nn.LayerNorm(output_layer_size))

    def compute_attentions(self, out: Tensor):
        # batches x ntaxa x seqlength x channels

        B, N, L, C = out.shape
        indices = make_indices(B, N, directed=False).to(out.device)

        for i in range(self.n_blocks):
            # AXIAL ATTENTIONS BLOCK
            # ----------------------
            # ROW ATTENTION
            # is flatten(0, 1).flatten(1, 2) AND .reshape((batch_size*n_seqs, -1)) equivalent?
            out = out.reshape(B * N, L * C)

            out = F.leaky_relu(out)  # DO WE NEED THIS?
            out = self.gats[i](out, indices)

            # does reshape (batch_size,n_seqs,hidden_channels,-1) give same result?
            out = out.reshape((B, N, -1, C))
            # out = (out
            #        .unflatten(0, (batch_size, n_seqs))
            #        .unflatten(-1, (self.hidden_channels, - 1)))

            # COLUMN ATTENTION
            # expects batches x taxa x seqlen x features
            att, a = self.column_attentions[i](out)

            out = att + out
            out = self.layernorms[i](out)  # layernorm
            out = self.fNNs[i](out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) + out
            if hasattr(self, "downsampling_layers") and not i % 2:
                out = self.downsampling_layers[i // 2](out.permute(0, 3, 1, 2)).permute(
                    0, 2, 3, 1
                )
                B, N, L, C = out.shape

            # batches x ntaxa x seqlength x channels

            # FEEDFORWARD
            # TODO:
        return out

    # def compute_embeddings(self, x):

    #     out = self.embed(x.long()).permute(0, 3, 1, 2, )
    #     out = self.block_1_1(out)
    #     out = (self
    #            .norm(out.transpose(-1, -3))
    #            .permute(0, 2, 1, 3)
    #            )  # layernorm
    #     # batches x ntaxa x seqlength x channels
    #     out = self.pool(out)
    #     return out

    def forward(self, x: torch.Tensor):
        # TODO: extend this so we can make a Q function that takes an (alignment, distance_matrix) pair; concat to new col in out after block_1_1
        # x hash shape (batches x) ntaxa x seqlength
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        # dmat has shape (batches x) ntaxa x ntaxa
        out = self.compute_embeddings(x)  # BxNxLxC
        out = self.compute_attentions(out)

        # out = norm(out)
        # out = out.reshape((n_seqs, batch_size, -1, seq_length))
        if self.format == "distance":
            out = torch.stack([torch.pdist(x, 2) for x in out])
        else:
            out = torch.stack([x.cov() for x in out])
        # out = -out.mean(-1)  # RELU followed by -, since log-likelihood <=0
        # TODO: replace with:
        # out = out.mean(-1)  # .relu()

        return out


class AxialGCN(AttentionNet):
    def __init__(
        self, k=4, format: Literal["distance", "covariance"] = "distance", **kwargs
    ):
        """interleave GAT and column attention layers.
        stride must evenly divide hidden_channels*compressed_length since we will do a lot of reshaping

        Args:
            compressed_length (int, optional): compress to (hidden_channels x) compressed_length with adaptive pooling before passing to GAT. Defaults to 200.
            base (int, optional): decrease size by this amount after each GAT. Defaults to 1.
        """
        super().__init__(**kwargs)
        # TODO: like attentionnet, but use gatconv instead of row attention layers
        self.format = format
        # TODO: write new attention layer that looks at each s
        # must reshape to (...,seq_length,hidden_channels)
        # self.pool = nn.AdaptiveAvgPool2d((compressed_length, None))
        self.k = k
        #  nn.Flatten()
        # del self.layernorms  # don't need this since we're flattening all channels
        # del self.row_attentions
        if self.stride > 1:
            del self.downsampling_layers
        del self.fNNs

        hidden_channels = kwargs["hidden_channels"]
        fNN_channels = kwargs["hidden_channels"]
        # TODO: use pyg PositionalEncoding layer
        attention_heads = kwargs["attention_heads"]
        dropout = kwargs.get("dropout", 0)
        # TODO use EdgeConv we wrote rather than the GATv2Conv extension to handle n-dim tensors

        self.gcns = nn.ModuleList()
        for i in range(kwargs["n_attention_layers"] // 2):
            seq_edge_mapper = nn.Sequential(
                nn.GELU(),
                nn.Conv1d(
                    in_channels=2 * hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                ),  # if i % 2 else 1)
                nn.GELU(),
                nn.Conv1d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    stride=1,
                ),
            )
            self.gcns.append(EdgeBlockConv(seq_edge_mapper, aggr="mean", cat_dim=-2))

            # TODO: decide whether to flatten row- or column-major, and how to downsample appropriately over seq_length NOT channels
            # self.gat_layer_norms.append(nn.LayerNorm(output_layer_size))

    def compute_attentions(self, out: Tensor):
        # batches x ntaxa x seqlength x channels

        B, N, L, C = out.shape
        batches = torch.arange(B, device=out.device).repeat_interleave(N)

        for i in range(self.n_blocks):
            # AXIAL ATTENTIONS BLOCK
            # ----------------------
            # ROW ATTENTION
            # is flatten(0, 1)  equivalent?
            # does reshape (batch_size,n_seqs,hidden_channels,-1) give same result?
            # out = out.reshape((B, N, -1, C))
            # out = (out
            #        .unflatten(0, (batch_size, n_seqs))
            #        .unflatten(-1, (self.hidden_channels, - 1)))
            if hasattr(self, "row_attentions"):

                att, a = self.row_attentions[i](out)
                # row attention+residual connection
                out = att + out
                out = self.layernorms[i](out)  # layernorm

            # COLUMN ATTENTION
            # expects batches x taxa x seqlen x features
            # column attention+residual connection
            if hasattr(self, "column_attentions"):

                att, a = self.column_attentions[i](out)

                out = att + out
                out = self.layernorms[i](out)  # layernorm
                # out = self.fNNs[i](out.permute(0, 3, 1, 2)).permute(
            #     0, 2, 3, 1)+out  # todo: incorporate into edge_mapper

            if not i % 2:
                B, N, L, C = out.shape

                out = out.reshape(B * N, L, C)
                edges = pool.knn_graph(
                    x=out.flatten(1, -1),
                    k=self.k,
                    num_workers=8,
                    loop=True,  # this just tells knn_graph to look for k+1 nn and not exclude self
                    batch=batches,
                )
                edges = to_undirected(edges)

                out = self.gcns[i // 2](out.transpose(-1, -2), edges).transpose(-1, -2)
                out = out.unflatten(0, (B, N))

            # batches x ntaxa/npairs x seqlength x channels

            # FEEDFORWARD
            # TODO:

        return out

    def compute_embeddings(self, x):
        B, N, L = x.shape
        out = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )
        out = self.block_1_1(out)

        if hasattr(self, "pos_encoding"):
            out = self.pos_encoding(out)

        # out = torch.matmul(self.seq2pair(N).to(out.device), out)
        # TODO: does S.t() convert pairs to seqs?

        out = self.norm(out.transpose(-1, -3)).permute(0, 2, 1, 3)  # layernorm
        # batches x ntaxa/npairs x seqlength x channels
        # out = self.pool(out)
        return out

    def forward(self, x: torch.Tensor):
        # TODO: extend this so we can make a Q function that takes an (alignment, distance_matrix) pair; concat to new col in out after block_1_1
        # x hash shape (batches x) ntaxa x seqlength
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        # dmat has shape (batches x) ntaxa x ntaxa
        out = self.compute_embeddings(x)  # BxNxLxC
        out = self.compute_attentions(out)
        out = self.pwFNN(out.permute(0, 3, 1, 2)).squeeze()
        # out = norm(out)
        # out = out.reshape((n_seqs, batch_size, -1, seq_length))

        if self.format == "distance":
            out = torch.stack([torch.pdist(x, 2) for x in out])
        else:
            out = torch.stack([x.cov() for x in out])

        # out = -out.mean(-1)  # RELU followed by -, since log-likelihood <=0
        # TODO: replace with:
        # out = out.mean(-1)  # .relu()

        return out


class SeqGAT(SeqNet):
    def __init__(
        self,
        *args,
        gat_layer_sizes=[100, 50, 25],
        edge_dim: int = None,
        dropout=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        attention_heads = kwargs["attention_params"]["attention_heads"]

        # TODO: use pyg PositionalEncoding layer

        # TODO use EdgeConv we wrote rather than the GATv2Conv extension to handle n-dim tensors
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, gat_layer_sizes[0]))

        self.gats = nn.ModuleList()

        for i, (input_layer_size, output_layer_size) in enumerate(
            zip(gat_layer_sizes, gat_layer_sizes[1:])
        ):
            if output_layer_size == 0:
                raise ValueError(
                    "base, n_attention_layers must be specified s.t. final gat layer is not empty"
                )
            self.gats.append(
                GATv2Conv(
                    input_layer_size * (i and attention_heads or 1),
                    output_layer_size,
                    heads=attention_heads,
                    edge_dim=edge_dim,
                    fill_value=0,
                    bias=True,
                    concat=True,
                    share_weights=True,  # undirected graph
                    dropout=dropout,
                )
            )
        self.pwFNN = nn.Sequential(
            nn.Conv1d(
                in_channels=attention_heads * output_layer_size,
                out_channels=output_layer_size,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(dropout),
            nn.Softplus(),
        )

    def forward(self, x, dmat=None):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        if x.dim() < 3:
            x = x.unsqueeze(0)
        if not self.transfer:
            x = self.compute_embeddings(x)
        x = self.attentions(x).transpose(1, 2)
        B, N, D, L = x.shape
        # TODO: does this work better than avg over sites?
        # x=rearrange(x, 'b n d l -> b n (l d)')
        x = self.pooling(x)  # .squeeze()
        # B D N L -> B N D' L' -> B*N D'*L', use if we want to  have a 1D conv with kernel = k*D'
        x = rearrange(x, "b n d l -> (b n) (l d)")
        # x = rearrange(x, 'b n l -> (b n) l')

        indices = make_indices(B, N, directed=False, device=x.device)

        if dmat is not None:
            dmat = dmat.repeat((1, 2))  # undirected network
            dmat = dmat.ravel()
            for gat in self.gats:
                x = gat(x, indices, edge_attr=dmat.to(x.device))
        else:
            for gat in self.gats:
                x = gat(x, indices)
        # x.unflatten(0, (B, N)).transpose(2, 1)  # B D N
        x = rearrange(x, "(b n) d -> b d n", b=B, n=N)
        x = self.pwFNN(x).transpose(1, 2)  # B N D'
        return self.compute_distances(x)


class AttentionGAT(AttentionNet):
    def __init__(
        self,
        *args,
        compressed_length=50,
        base=1,
        n_gat_layers=1,
        edge_dim: int = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        attention_heads = kwargs["attention_heads"]
        dropout = kwargs["dropout"]
        # TODO: use pyg PositionalEncoding layer

        # TODO use EdgeConv we wrote rather than the GATv2Conv extension to handle n-dim tensors
        self.pooling = nn.AdaptiveAvgPool1d(compressed_length)

        output_layer_size = compressed_length // base
        self.gats = nn.ModuleList(
            [
                GATv2Conv(
                    compressed_length,
                    output_layer_size,
                    heads=attention_heads,
                    edge_dim=edge_dim,
                    fill_value=0,
                    bias=True,
                    share_weights=True,  # undirected graph
                    dropout=dropout,
                )
            ]
        )

        for i in range(1, n_gat_layers):
            input_layer_size = attention_heads * (compressed_length // base**i)
            output_layer_size = compressed_length // base ** (i + 1)
            if output_layer_size == 0:
                raise ValueError(
                    "base, n_attention_layers must be specified s.t. final gat layer is not empty"
                )

            self.gats.append(
                GATv2Conv(
                    input_layer_size,
                    output_layer_size,
                    heads=attention_heads,
                    edge_dim=edge_dim,
                    fill_value=0,
                    bias=True,
                    share_weights=True,  # undirected graph
                    dropout=dropout,
                )
            )
            # self.gat_layer_norms.append(nn.LayerNorm(output_layer_size))
            # TODO: try using edge attention weights as output OR add edge GCN
        input_layer_size = attention_heads * output_layer_size
        self.fc = nn.Sequential(
            # nn.LeakyReLU(),
            nn.LayerNorm(input_layer_size),
            nn.Linear(
                in_features=input_layer_size,
                out_features=input_layer_size // 2,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Linear(in_features=input_layer_size // 2, out_features=1, bias=True),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, dmat: torch.Tensor = None):
        # TODO: extend this so we can make a Q function that takes an (alignment, distance_matrix) pair; concat to new col in out after block_1_1
        # x hash shape (batches x) ntaxa x seqlength
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        # dmat has shape (batches x) ntaxa x ntaxa
        batch_size, n_seqs, seq_length = x.shape
        n_pairs = n_seqs * (n_seqs) // 2
        out = self.compute_embeddings(x)
        out = self.compute_attentions(out)
        # batches x n_pairs x n_features x compressed_seq_len
        out = out.permute(0, 2, 1, 3)
        # TODO how to get from pairs back to feature vec for each taxon?
        # out = out.reshape((batch_size*n_seqs, , -1))
        out = self.pooling(out)

        indices = make_indices(batch_size, n_seqs, directed=False)
        if dmat is not None:
            dmat = dmat.repeat((1, 2))  # undirected network
            dmat = dmat.ravel()
            for gat in self.gats:
                out = gat(out, indices.to(out.device), edge_attr=dmat.to(out.device))
        else:
            for gat in self.gats:
                out = gat(out, indices.to(out.device))
        # out = out.reshape((n_seqs, batch_size, -1, seq_length))
        out = self.fc(out).reshape((batch_size, n_seqs))
        # out = -out.mean(-1)  # RELU followed by -, since log-likelihood <=0
        return out


class QFunctionAttentionGAT(AttentionGAT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, softplus=False, edge_dim=1, **kwargs)

    def forward(self, x: torch.Tensor, dmat: torch.Tensor):
        # TODO: extend this so we can make a Q function that takes an (alignment, distance_matrix) pair; concat to new col in out after block_1_1
        # x hash shape (batches x) ntaxa x seqlength
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)
        # dmat has shape (batches x) ntaxa x ntaxa
        # AttentionGAT is always positive, but LL is always negative
        out = -super().forward(x, dmat)
        out = out.mean(-1) + self.baseline  # scaling  # .relu()

        return out


class QFunctionAttention(AttentionNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, softplus=False, **kwargs)
        hidden_channels = kwargs["hidden_channels"]
        self.dist_embed = nn.Conv1d(
            in_channels=1, out_channels=hidden_channels, kernel_size=1, stride=1
        )
        self.pos_encoding = PositionalEncoding(hidden_channels, max_len=1000)

    def forward(self, x, dmat):
        # 2d convolution that gives us the features in the third dimension (i.e. initial embedding of each amino acid)

        batch_size, ntaxa, seq_length = x.shape

        out = self.embed(x.long()).permute(
            0,
            3,
            1,
            2,
        )
        out = self.block_1_1(out)
        if hasattr(self, "pos_encoding"):
            out = self.pos_encoding(out)

        out = self.norm(out.transpose(-1, -3)).transpose(-1, -3)  # layernorm

        # batch_size x features x nb_pairs x 1
        dmat = self.dist_embed(dmat[:, None, :])[..., None]
        out = torch.matmul(self.seq2pair(ntaxa, out.device), out)

        # distance will always be in the first position
        out = torch.cat([dmat, out], dim=-1)

        # from here on the tensor has shape (batch_size,features,nb_pairs,seq_len+1), all the transpose/permute allow to apply layernorm
        # and attention over the desired dimensions and are then followed by the inverse transposition/permutation of dimensions

        out = self.compute_attentions(out)

        # after this last convolution we have (batch_size,1,nb_pairs,seq_len) ->  (batch_size,1,nb_pairs,1)
        # Attention is always positive, but LL is always negative
        out = -self.pwFNN(out)
        # averaging over positions AND pair and removing the extra dimensions we finally get (batch_size,1)
        out = out.view(batch_size, -1).mean(1) + self.baseline  # scaling
        return out


class DoubleBlockConvNetwork(nn.Module):
    def __init__(
        self,
        kernel: int = 200,
        stride: int = 100,
        stride_1D: int = 1,
        k: int = 3,
        n_graph_nets: int = 1,
        kernel_size_1D: int = 3,
        proj_kernel_size: int = 1,
        #  stride_1D: int = 2,
        site_conv_layers: list = [1024] * 4,
        taxon_conv_layers: list = None,
        char_embedding_dim: int = EMBED_DIM,
        dropout: float = 0.1,
        num_layers: int = 1,
        n_rounds: int = 1,
        batch_norm: bool = False,
        output_layers: list = [256] * 3,
        output="metric",
        device="cpu",
        **kwargs,
    ):
        """Char Embedding --> CNN Embedding --> [[taxa GCN] x N --> subsequence GCN --> projection NN] x M  --> MetricDecoder
        Each graph computes a new CNN along the input (must have stride 1 since all graph outputs are concatenated)

        Args:
            kernel (int, optional): size of char embed kernel. Defaults to 200.
            stride (int, optional): stride over alignment input. Defaults to 100.
            k (int, optional): k-nearest neighbor graph for message passing. Defaults to 3.
            n_graph_nets (int, optional): graph nets to pass through. Defaults to 1.
            kernel_size_1D (int, optional): edge feature conv. Defaults to 3.
            proj_kernel_size (int, optional): final feature conv. Defaults to 1.
            stride_1D (int, optional): stride. Defaults to 2.
            site_conv_layers (list, optional): last layer must have 1/2 the features as 1st. Defaults to [1024]*4.
            taxon_conv_layers (list, optional): unused. Defaults to [1024]*4.
            char_embedding_dim (int, optional): embed each character in n-dim Euclidean space. Defaults to EMBED_DIM.
            dropout (float, optional): dropout between layers. Defaults to .1.
            num_layers (int, optional): unused. Defaults to 1.
            n_rounds (int, optional): number of edge+site conv iterations. Defaults to 1.
            batch_norm (bool, optional): do not use. Defaults to False.
            output_layers (list, optional): unused. Defaults to [256]*3.
            output (str, optional): metric or covariance. Defaults to 'metric'.
            device (str, optional): all layers EXCEPT char_embed will be on device. Defaults to 'cpu'.
        """
        from torch_geometric.nn import EdgeConv

        self.debug = False
        # if num_layers > 1 and in_channels != output_channels:
        #     raise ValueError(
        #         'input dim must match output dim for multilayer GAT')
        super().__init__()
        # self.device = device
        self.k = k
        self.latent_dim = site_conv_layers[0] // 2
        self.embedding_dim = site_conv_layers[-1]
        self.n_graph_nets = n_graph_nets
        self.n_rounds = n_rounds

        self.embed = nn.Embedding(num_embeddings=22, embedding_dim=char_embedding_dim)

        self.block_1_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=char_embedding_dim,
                out_channels=self.latent_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        # embedding_network = nn.Sequential(*make_conv_net(
        #     [char_embedding_dim, self.latent_dim],
        #     kernel=kernel, stride=stride, dropout=dropout),
        #     nn.ELU()
        # )
        self.norm = nn.LayerNorm(self.latent_dim)

        graph_embed_dim = self.embedding_dim * n_graph_nets + self.latent_dim
        if taxon_conv_layers is None:
            taxon_conv_layers = [graph_embed_dim // 2, graph_embed_dim // 4]
        for r in range(n_rounds):
            for i in range(2):
                graph_nets = nn.ModuleList()
                for _ in range(n_graph_nets):
                    # must have same input and output dim
                    edge_mapper = make_conv_net(
                        site_conv_layers,
                        kernel_size_1D,
                        stride=stride_1D,
                        batch_norm=batch_norm,
                        nonlinearity=nn.ELU,
                        dropout=dropout,
                    )
                    site_graph = EdgeBlockConv(edge_mapper, aggr="mean")
                    graph_nets.append(site_graph)
                self.add_module(f"r{r}_graph_nets_{i}", graph_nets)
                seq_edge_mapper = make_conv_net(
                    taxon_conv_layers,
                    # [graph_embed_dim*2, *
                    #  taxon_conv_layers, self.latent_dim],
                    kernel=1,
                    stride=1,
                    nonlinearity=nn.ELU,
                    batch_norm=batch_norm,
                    dropout=dropout,
                )

                seq_graph = EdgeBlockConv(seq_edge_mapper, aggr="mean")
                self.add_module(f"r{r}_seq_graph{i}", seq_graph)

            # conv_layer_sizes = [graph_embed_dim // 2 **
            #                     i for i in range(2, 4)]+[self.latent_dim]
            conv_layer_sizes = [self.latent_dim] * 2
            projection_layer = make_conv_net_2d(
                conv_layer_sizes,
                kernel=proj_kernel_size,
                stride=max(1, proj_kernel_size // 2),
                batch_norm=batch_norm,
                dropout=dropout,
            )
            self.add_module(f"r{r}_projection", projection_layer)

        # self.add_module('unfold', nn.Unfold(
        #     kernel_size=(kernel, 1), stride=stride))

        if output == "metric":
            self.add_module("output_layer", MetricDecoder(clip=1e9, as_list=True))
        elif output == "covariance":
            self.add_module("output_layer", CovarianceDecoder(clip=1e9, as_list=False))

        # self.to(self.device)
        # self.char_embed.to('cpu')

    def __repr__(self):
        # +f"\ngraph layers:{', '.join(map(str,self.graph_layers))}"
        return super().__repr__()
        # self.set_devices(device)

    def apply_graph(self, graph, x, batches=None):
        edges = pool.knn_graph(
            x=x.flatten(1), k=self.k, loop=True, batch=batches
        )  # BlockEdgeConv
        x = graph(x, edge_index=edges)
        return x

    def forward(self, x, output=True):
        """input x shape is [batches, ntaxa,ngenes, alignment_length]"""
        B, N, L = x.shape
        # edge_index = make_indices(B,N, directed=False)
        batches = make_batch_indices(N, B, x.device)  # TODO: fix this function
        # x = x.long().flatten(0, 1)
        if not isinstance(x, torch.Tensor):
            x = torch.cat(x)
        out = self.embed(x.long())  # .permute(0, 3, 1, 2, )
        out = self.block_1_1(out.transpose(1, -1)).transpose(1, -1)
        out = self.norm(out).transpose(-1, -2)  # layernorm
        out = out.flatten(0, 1)

        with torch.cuda.amp.autocast(enabled=False):
            for r in range(self.n_rounds):
                # convolve similar taxa
                output_list = [out]
                for net_no, graph in enumerate(getattr(self, f"r{r}_graph_nets_0"), 1):
                    x = output_list[-1]
                    x = self.apply_graph(graph, x, batches)
                    output_list.append(x)
                out = torch.cat(output_list, 2)
                # convolve similar sites BY ALIGNMENT (could also do BY SEQUENCE but this would be slower)
                out = out.unflatten(0, (B, N))
                graph = getattr(self, f"r{r}_seq_graph0")
                seq_knn_graphs = [
                    pool.knn_graph(x.T, k=self.k, loop=True) for x in out.flatten(1, 2)
                ]
                output_list = [out]
                for i, x in enumerate(out):
                    # edges = pool.knn_graph(
                    #     seq_features[batches[i]].T, k=self.k)
                    edges = seq_knn_graphs[i]
                    output_list.append(graph(x.permute(2, 1, 0), edges).T)
                out = torch.stack(output_list, 0)

                # convolve similar taxa
                output_list = [out.flatten(0, 1)]
                for net_no, graph in enumerate(getattr(self, f"r{r}_graph_nets_1"), 1):
                    x = output_list[-1]
                    x = self.apply_graph(graph, x, batches)
                    output_list.append(x)
                out = torch.cat(output_list, -1)

                out = out.unflatten(0, (B, N))
                graph = getattr(self, f"r{r}_seq_graph0")
                seq_knn_graphs = [
                    pool.knn_graph(x.T, k=self.k, loop=True) for x in out.flatten(1, 2)
                ]
                output_list = [out]
                for i, x in enumerate(out):
                    # edges = pool.knn_graph(
                    #     seq_features[batches[i]].T, k=self.k)
                    edges = seq_knn_graphs[i]
                    output_list.append(graph(x.permute(2, 1, 0), edges).T)
                out = torch.stack(output_list, 0)
                graph = getattr(self, f"r{r}_projection")
                out = graph(out.transpose(1, 2))

        out = torch.stack(self.output_layer(out.transpose(1, 2).flatten(2)))
        return out
