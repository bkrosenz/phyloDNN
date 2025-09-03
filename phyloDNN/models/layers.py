from functools import lru_cache
from multiprocessing.dummy import Value
from einops import rearrange, repeat
import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for 1D sequences"""

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        # self.encoding = self.encoding.reshape(1, 1, max_len, d_model)
        self.register_buffer("pe", encoding.t())
        self.parameter = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # x: (batch_size, features, nb_pairs, seq_len)
        # out: (batch_size, features, nb_pairs, seq_len)
        length = x.shape[-1]
        if length > self.pe.shape[1]:
            raise ValueError(
                f"Input length {length} exceeds max_len {self.pe.shape[1]}, must redefine PositionalEncoding layer"
            )
        x = x + self.pe[:, :length] * self.parameter
        return x


# from torch._dynamo import allow_in_graph

# allow_in_graph(rearrange)

# borrowed from lucidrains
# https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py#L21


def relative_to_absolute(q):
    """
    Converts the dimension that is specified from the axis
    from relative distances (with length 2*tokens-1) to absolute distance (length tokens)
      Input: [bs, heads, length, 2*length - 1]
      Output: [bs, heads, length, length]
    """
    from einops import rearrange

    b, h, l, *_, device, dtype = *q.shape, q.device, q.dtype
    dd = {"device": device, "dtype": dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
    flat_x = rearrange(x, "b h l c -> b h (l c)")
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1) :]
    return final_x


def rel_pos_emb_1d(q, rel_emb):
    """
    Same functionality as RelPosEmb1D

    Args:
        q: a 4d tensor of shape [batch x seqlen x nheads x pairs x nfeatures//nheads]
        rel_emb: a 2D or 3D tensor
        of shape [ 2*tokens-1 , dim] or [ heads, 2*tokens-1 , dim]
    """
    emb = torch.einsum("b t h p d, h d r -> b t h p r", q, rel_emb)
    return relative_to_absolute(emb)


class RelPosEmb(nn.Module):
    def __init__(self, dim_head, max_len=1000, heads=1):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: the number of the tokens of the seq
            dim_head: the size of the last dimension of q

            heads: if None representation is shared across heads.
            else the number of heads must be provided
        """
        super().__init__()
        # pe = torch.zeros(max_len, dim_head)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(
        #     0, dim_head, 2).float() * (-math.log(10000.0) / dim_head))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # # final shape: 1 x <dim_head> x 1 x<max_len>

        # # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # # Used for tensors that need to be on the same device as the module.
        # # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        # self.register_buffer('pe', pe, persistent=False)
        # self.pe_weights = nn.Parameter(torch.randn(max_len, dim_head))

        # looks back by max_len positions
        self.Er = nn.Parameter(torch.randn(heads, max_len, dim_head))

    def forward(self, q):
        _, L, *_ = q.shape
        H, N, Dh = self.Er.shape
        # start = max(self.max_len, self.max_len - seq_len)
        # Er_t = torch.outer(self.pe_weights, self.pe)
        # Er_t.shape = (d_head, seq_len)
        Er = torch.nn.functional.pad(self.Er, (0, 0, L - N, 0))  # TODO: convolve?
        QEr = torch.matmul(q, Er.transpose(-1, -2))
        padded = torch.nn.functional.pad(QEr, (1, 0))
        Srel = padded.reshape(H, L + 1, L)[:, 1:, :]
        mask = torch.triu(torch.ones(L, L), diagonal=L - N).bool()
        Srel = Srel.masked_fill(mask, 0)

        # a la https://gudgud96.github.io/2020/04/01/annotated-music-transformer/

        # TODO: figure out how to linearize this - mask the \phi(K) sums?
        # softmax(Q(K^T+R^T)/\sqlt(d))V

        # Srel = self.skew(QEr)
        return Srel

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len_Er)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len_Er)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class PermutationEquivariantLayer(nn.Module):
    """computes \lambda I + \mu 11^T  output has same shape as input, where sum is over pairs dimension rather than sites"""

    def __init__(self, heads: int = 1):
        """Heads parameter allows subsets of channels to be grouped together.  Dont need this to have an invariant version"""
        super().__init__()
        self.heads = heads
        self.lam = nn.Linear(heads, heads)
        self.mu = nn.Linear(heads, heads)

    def forward(self, x: torch.Tensor):
        """batches x channels x n_pairs x 2*n_sites"""
        x = rearrange(x, "b (c h) p l -> b c p l h", h=self.heads)
        x = self.lam(x) + self.mu(x.sum(-3, keepdim=True)).expand(x.shape)
        return rearrange(x, "b c p l h -> b (c h) p l")


class EquivariantLayer(nn.Module):
    """computes \kappa I + \lam [0, I; I, 0] + \mu [0,11^T;11^T,0] + \nu [11^T,0;0,11^T] , output has same shape as input"""

    def __init__(self, invariant: bool = False, heads: int = 1, activation=F.sigmoid):
        """Heads parameter allows subsets of channels to be grouped together"""
        super().__init__()

        self.activation = activation
        self.invariant = invariant
        self.heads = heads
        self.kappa = nn.Linear(heads, heads)
        if not invariant:
            self.lam = nn.Linear(heads, heads)
            self.mu = nn.Linear(heads, heads)
            self.nu = nn.Linear(heads, heads)
        # if invariant:
        #     self.linear = nn.Linear(2, heads)
        # else:
        #     self.linear = nn.Linear(3, heads)

    def forward(self, x: torch.Tensor, ix: torch.Tensor):
        """batches x channels x n_pairs x 2*n_sites"""
        length = x.shape[-1]
        d = length // 2
        x = rearrange(x, "b (c h) ... -> b c ... h", h=self.heads)

        # TODO: use masked tensors to expand sums to proper shape

        if self.invariant:
            x = self.kappa(x.sum(dim=-2)) #x_sums[..., 0, :]) + self.lam(x_sums[..., 1, :])
            # return x.mean(-1) # average over heads
        else:

            mask = torch.zeros(length, dtype=torch.long, device=x.device)
            mask[:d] = 1

            x_sums = torch.cat(
                (
                    x[..., :d, :].sum(-2, keepdim=True),
                    x[..., d:, :].sum(-2, keepdim=True),
                ),
                dim=-2,
            )
            x = (
                self.kappa(x)
                + self.lam(x.index_select(dim=-2, index=ix))
                + self.mu(x_sums).index_select(dim=-2, index=mask)
                + self.nu(x_sums).index_select(dim=-2, index=1 - mask)
            )
        return rearrange(x, "b c ... h -> b (c h) ...")
        # if self.invariant:
        #     # equivalent to \lambda I .sum() + \mu 11^T .sum() + \nu [I, 0; 0, I] .sum()
        #     x = self.linear(
        #         torch.cat([x[..., :d].sum(-1,keepdim=True), x[...,d:].sum(-1,keepdim=True)], dim=-1))
        # else:
        #     # print(x.shape, x.sum(-1,keepdim=True).expand(x.shape).shape, torch.cat([x[...,d:], x[...,:d]] ).shape )
        #     x=torch.stack(
        #         [x, x.sum(-1,keepdim=True).expand(x.shape), torch.cat([x[...,d:], x[...,:d]], dim=-1) ],
        #         dim = -1)
        #     x = self.linear(x)


class LinearSplit(nn.Linear):
    """Linear layer with split weights and biases for each half of a concatenated x1|x2 sequence:
    y = [w1x1|w1x2]+[w2x2|w2x1]"""

    def __init__(self, in_features, out_features, heads=1, bias=True):
        super().__init__(in_features, out_features * heads, bias=bias)
        self.heads = heads

    def forward(self, x):
        # x.shape = (batch_size, in_features)
        # w.shape = (out_features * heads, in_features)
        # b.shape = (out_features * heads)
        # return F.linear(x, self.weight.view(self.heads, -1), self.bias.view(self.heads))
        x = x.view(x.shape[0], self.heads, -1)
        return F.linear(x, self.weight.view(-1), self.bias.view(-1))


class AxialAttention(nn.Module):

    def __init__(
        self,
        h_dim,
        n_heads,
        dropout=0.0,
        eps=1e-6,
        # pos_embed=False,
        field_size=None,
        unfold=False,
        with_q=False,
        kq_dim: int | None = None,
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
        self.n_heads = n_heads
        self.q_net = nn.Linear(h_dim, kq_dim or h_dim)
        self.k_net = nn.Linear(h_dim, kq_dim or h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)
        self.elu = nn.ELU()
        self.eps = eps
        self.field_size = field_size

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.dropout = dropout
        self.proj_drop = nn.Dropout(dropout)
        self.with_q = with_q
        # if pos_embed:
        #     self.pos_embed = RelPosEmb(dim_head=d_head, heads=n_heads)

    @lru_cache
    def get_mask(self, S, L, device):
        mask = torch.zeros((S, L), dtype=torch.bool, device=device)
        mask[torch.tril_indices(*mask.shape, offset=self.field_size // 2)] = True
        mask = mask + mask.t()
        mask = mask[None, None, None, ...]
        return mask

    def forward(self, x: torch.Tensor):
        # shape (col attn) : batch x seqlen x pairs x features
        # row attn: (batch_size,nb_pairs,seq_len,features)
        Bs = x.shape[0]
        M, T, C = x.shape[-3:]
        N, D = self.n_heads, C // self.n_heads

        # shape: batch x seqlen x nheads x pairs x nfeatures//nheads
        # TODO: allow different q/k and v dims
        q = self.q_net(x).view(Bs, M, T, N, D).transpose(2, 3)
        k = self.k_net(x).view(Bs, M, T, N, D).transpose(2, 3)
        v = self.v_net(x).view(Bs, M, T, N, D).transpose(2, 3)

        # q = self.elu(q)+1
        # k = self.elu(k)+1
        if self.field_size is not None:
            mask = self.get_mask(q.size(-2), k.size(-2), x.device).expand(
                Bs, M, N, -1, -1
            )
        else:
            mask = None
        V = nn.functional.scaled_dot_product_attention(q, k, v, mask)

        V = V.transpose(2, 3).contiguous().view(Bs, -1, T, N * D)

        a = None
        out = self.proj_drop(self.proj_net(V))
        return out, a


class KernelAxialMultiAttention(nn.Module):
    def __init__(
        self,
        h_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
        pos_embed: bool = False,
        field_size: int | None = None,
        unfold: bool = False,
        is_causal: bool = False,
        with_q: bool = False,
        share_qk: bool = False,
        sdpa: bool = False,
        masking: bool = False,
    ):
        """axial attention with linear softmax approximation.  with_q fixes bug in Phyloformer

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
        self.sdpa = sdpa
        self.is_causal = is_causal
        self.heads = n_heads
        self.h_dim = h_dim
        self.share_qk = share_qk
        self.qkv_net = nn.Linear(h_dim, h_dim * (3 - share_qk))
        self.elu = nn.ELU()
        self.eps = eps
        self.field_size = field_size
        self.unfold = unfold
        if field_size is not None:
            if unfold:
                self.unfold = nn.Unfold((field_size, 1, 1), stride=field_size)
            else:
                self.pool_ktv = nn.AvgPool3d(
                    (field_size, 1, 1),
                    stride=1,
                    padding=(field_size // 2, 0, 0),
                    divisor_override=1,
                )
                self.pool_k = nn.AvgPool3d(
                    (1, field_size, 1),
                    stride=1,
                    padding=(0, field_size // 2, 0),
                    divisor_override=1,
                )

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.dropout = dropout
        self.proj_drop = nn.Dropout(dropout)
        self.with_q = with_q
        # if pos_embed:
        #     self.pos_embed = RelPosEmb(dim_head=d_head, heads=n_heads)

        if masking:
            self.mask_weights = nn.Parameter(torch.rand(2), requires_grad=True)

    def forward(self, x, mask: torch.Tensor = None):
        # shape (col attn) : batch x seqlen x pairs x features, mask shape nb_pairs
        # row attn: (batch_size,nb_pairs,seq_len,features), mask shape seqlen
        # col attn: (batch_size,seq_len,nb_pairs,features)
        Bs, M, T, C = x.shape
        D = C // self.heads
        h = self.heads
        # shape: batch x seqlen x nheads x pairs x nfeatures//nheads for col
        # shape: batch x nb_pairs x nheads x seqlen x nfeatures//nheads for row
        # TODO: allow different q/k and v dims
        if self.share_qk:
            q, v = self.qkv_net(x).chunk(2, dim=-1)
            q, v = map(
                lambda t: rearrange(t, "... l (h d) -> ... h l d", h=self.heads), (q, v)
            )
            k = q
        else:
            q, k, v = self.qkv_net(x).chunk(3, dim=-1)
            q, k, v = map(
                lambda t: rearrange(t, "... l (h d) -> ... h l d", h=self.heads),
                (q, k, v),
            )

        if mask:
            mask = (mask < mask.max() * self.mask_weights[0]).view(1, 1, -1, 1)
            k = k * mask

        if self.sdpa:
            V = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout, is_causal=self.is_causal
            )
            # if self.is_causal:
            #     V=V+ F.scaled_dot_product_attention(q,k.flip(-2),v,dropout_p=self.dropout,is_causal=self.is_causal)

        q = self.elu(q) + 1
        if not self.share_qk:
            k = self.elu(k) + 1

        if self.field_size is not None:
            if self.unfold:

                k, q, v = map(self.unfold, k, q, v)
                V = V.transpose(2, 3).contiguous().view(Bs, -1, T, h * D)

                a = None
                out = self.proj_drop(self.proj_net(V))
                return out, a

                # unroll to make local attention
            else:
                # todo: use unroll to do this in blocks instead of sliding windows for LONG seqs
                k_roll = self.pool_k(k)
                # outer product, then sum in windows (don't need to unscale by field_size)
                KtV = torch.einsum("...x,...y->...xy", k, v)
                KtV = self.pool_ktv(KtV.flatten(end_dim=1)).unflatten(0, (Bs, M))
                V = torch.einsum("...d,...dm->...m", q, KtV)
                Z = 1 / (torch.einsum("...l,...l->...", q, k_roll))
                V = V * Z[..., None]
        else:
            KtV = k.transpose(-1, -2) @ v  # unroll to make local attention
            # shape: batch x nb_pairs x nheads x nfeatures//nheads x nfeatures//nheads for row
            Z = 1 / (
                q @ k.transpose(-1, -2).sum(dim=-1, keepdim=True) + self.eps
            )  # unroll k...sum to make local
            # shape: batch x nb_pairs x nheads x nfeatures//nheads x 1 for row

            if self.with_q:
                V = q @ KtV
                V = Z * V
            else:
                Z = Z.expand(Bs, M, h, T, D)
                # shape: batch x nb_pairs x nheads x seqlen x nfeatures//nheads for row

                V = Z @ KtV
        # shape: batch x nb_pairs x nheads x seqlen x nfeatures//nheads for row

        V = V.transpose(2, 3).contiguous().view(Bs, -1, T, h * D)
        # shape: batch x nb_pairs x seqlen x nfeatures for row

        a = None
        out = self.proj_drop(self.proj_net(V))
        return out, a


class CompressedMultiAttention(nn.Module):
    def __init__(
        self,
        h_dim: int,
        n_heads: int,
        latent_dim: int,
        dropout: float = 0.0,
    ):
        """compressed attention. Each layer has a set of learnable 'register' vectors that serve as queries.
        Args:
            h_dim (int ): must be divisible by n_heads
            n_heads (_type_): split channels into separate heads
            latent_dim (int): number of registers
            dropout (float, optional): _description_. Defaults to 0.0.
            pos_embed (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        d_head, remainder = divmod(h_dim, n_heads)
        if remainder:
            raise ValueError("incompatible `d_model` and `num_heads`")
        self.heads = n_heads
        self.h_dim = h_dim
        self.latent_dim = latent_dim

        self.kv_net = nn.Linear(h_dim, h_dim * 2)
        # self.q_net = nn.Linear(h_dim, h_dim)

        self.registers = nn.Parameter(torch.rand(latent_dim, h_dim), requires_grad=True)
        self.mha = nn.MultiheadAttention(
            embed_dim=h_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.elu = nn.ELU()
        self.proj_net = nn.Linear(h_dim, h_dim)

        self.dropout = dropout
        self.proj_drop = nn.Dropout(dropout)    

    def forward(self, x):
        # shape : (batch_size,features,nb_pairs,2*seq_len)
        B, D, N, S = x.shape
        out = []
        for xx in x.chunk(2, dim=-1):

            k, v = self.kv_net(xx.transpose(-1, -3)).chunk(
                2, dim=-1
            )  # (batch_size,seq_len,nb_pairs,features,)
            q = self.registers.expand(B*N, self.latent_dim, self.h_dim)
            k = rearrange(k, "b s n d -> (b n) s d")
            v = rearrange(v, "b s n d -> (b n) s d")
            attn_output, _ = self.mha(q, k, v, need_weights=False)
            out.append(attn_output)
        out = torch.cat(out, dim=-2)  # (batch_size*nb_pairs, 2*n_registers, features)
        out = self.proj_drop(self.proj_net(out))
        out = rearrange(out, "(b n) s d -> b d n s", b=B, n=N)
        return out


class LinearMultiAttention(nn.Module):

    def __init__(
        self,
        h_dim,
        n_heads,
        dropout=0.0,
        eps=1e-6,
        # pos_embed=False,
        n_output=None,
        with_q=False,
        sdpa=False,
        masking=False,
    ):
        """attention with linear softmax approximation.
        with_q fixes bug in Phyloformer.
        NOT self-attention: k values are n_output trainable registers.

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
        self.sdpa = sdpa
        self.heads = n_heads
        self.with_q = with_q
        self.n_output = n_output
        self.h_dim = h_dim
        self.kv_net = nn.Linear(h_dim, h_dim * 2)
        self.elu = nn.ELU()
        self.eps = eps
        self.q = nn.Parameter(
            torch.randn((n_output, h_dim))
            * int(max(1, math.sqrt(2 / (n_output + h_dim))))
        )  # Xavier?

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.dropout = dropout
        self.proj_drop = nn.Dropout(dropout)

        # if pos_embed:
        #     self.pos_embed = RelPosEmb(dim_head=d_head, heads=n_heads)

        if masking:
            self.mask_weights = nn.Parameter(torch.rand(2))

    def forward(self, x, mask=None):
        # row attn: (batch_size,nb_pairs,seq_len,features), mask shape seqlen
        #
        Bs, M, T, C = x.shape
        D = C // self.heads
        h = self.heads
        if mask:
            mask = (mask < mask.max() * self.mask_weights[0]).view(1, 1, -1, 1)
            # if mask.dtype==torch.bool:
            #     k=k*mask
            # elif mask.dtype==torch.float:
            #     k=k+mask
            k = k * mask

        # shape: batch x seqlen x nheads x pairs x nfeatures//nheads
        # TODO: allow different q/k and v dims
        k, v = self.kv_net(x).chunk(2, dim=-1)
        k, q, v = map(
            lambda t: rearrange(t, "... l (h d) -> ... h l d", h=self.heads),
            (k, self.q, v),
        )
        q = repeat(q, "h l d -> b p h l d", b=Bs, p=M)
        if self.sdpa:
            V = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout, is_causal=False
            )
            # if self.is_causal:
            #     V=V+ F.scaled_dot_product_attention(q,k.flip(-2),v,dropout_p=self.dropout,is_causal=self.is_causal)

        else:
            # q = self.elu(q)+1
            k = self.elu(k) + 1

            KtV = k.transpose(-1, -2) @ v  # unroll to make local attention

            Z = 1 / (
                q @ k.transpose(-1, -2).sum(dim=-1, keepdim=True) + self.eps
            )  # unroll k...sum to make local
            if self.with_q:
                V = q @ KtV
                V = V * Z
            else:
                Z = Z.expand(Bs, M, h, self.n_output, D)
                V = Z @ KtV

        V = V.transpose(2, 3).contiguous().view(Bs, -1, self.n_output, h * D)

        a = None
        out = self.proj_drop(self.proj_net(V))
        return out, a
