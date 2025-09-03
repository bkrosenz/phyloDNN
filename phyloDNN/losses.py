from numpy import dtype
from torch.cuda.amp import autocast
from .utils import array_to_mat, upper_triangular,squareform
import torch
__package__ = 'phyloDNN'


def s(x: torch.Tensor):
    n = torch.tensor(x.shape[0]).float()
    z = torch.floor(1+torch.log2(n))
    return n*z-2**z+1


@torch.jit.script
def log_sum_exp_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """compute differentiable approximation to \ell_\infty loss.
        batch is first dim, computes separately for each phylogeny in the batch.  

    Args:
        pred (torch.Tensor): predicted distance matrix
        true (torch.Tensor): true distance matrix

    Returns:
        torch.Tensor: _description_
    """
    diff = upper_triangular(pred-true).abs()
    return torch.logsumexp(diff, dim=-1)


@torch.jit.script
def kendalls_tau_loss(pred: torch.Tensor, true: torch.Tensor):
    '''rank taxa pairs most distant to least distant, count correct placements'''
    n = len(pred)
    x, y = torch.argsort(pred), torch.argsort(true)
    combs = torch.combinations(torch.arange(n), 2).T.to(pred.device)
    s0 = torch.sum(
        torch.index_select(x, 0, combs[0]) > torch.index_select(y, 0, combs[1])
    )
    s1 = torch.sum(
        torch.index_select(x, 0, combs[1]) < torch.index_select(y, 0, combs[0])
    )
    return 1 - (s0+s1) * 2 / (n*(n-1.))


def normalize_loss(loss):
    def new_loss(pred: torch.Tensor, true: torch.Tensor):
        '''must use sum reduction; with MSE, an n-batch of 1 n-graph has lower loss than an n-batch of 2 n/2-graphs.
        Therefore, we normalize by the number of actual edges in all the graphs in the batch (multiply by 2 because of symmetry).'''
        with autocast(enabled=False):
            return loss(pred.double(), true.double())/true.count_nonzero()*2
    return new_loss


smooth_l1_loss = normalize_loss(torch.nn.SmoothL1Loss(beta=2, reduction='sum'))

mse_loss = normalize_loss(torch.nn.MSELoss(reduction='sum'))

l1_loss = normalize_loss(torch.nn.L1Loss(reduction='sum'))


def rank_loss(pred: torch.Tensor, true: torch.Tensor):
    '''rank taxa pairs most distant to least distant, count correct placements'''
    return torch.argsort(pred).ne(torch.argsort(true)).sum().float() // s(true)


def msle_loss(output, target):
    '''truncated msle'''
    loss = torch.clamp(torch.mean(
        torch.abs(torch.log((1e-6 + output) / target))), 0, 1e10)
    return loss


def rank_l1(pred: torch.Tensor, true: torch.Tensor):
    '''rank taxa pairs most distant to least distant, sum distances'''
    r = torch.abs(torch.argsort(pred)-torch.argsort(true)
                  ).sum().float() / s(pred)
    return r  # .requires_grad_()


class LogDetLoss(torch.nn.Module):
    def __init__(
        self, eps=1e-3, format="distance", track_conditioning=False, radial=False
    ):
        """build LogDet-regularized loss with base loss (default=MSE)."""
        super().__init__()
        self.eps = 1e-5
        self.as_covariance = format == "covariance"
        self.radial = radial
        self.track_conditioning = track_conditioning

    def conditioning(self, mat):
        if self.radial:
            mat = (-(mat**2)).exp()
        cond = torch.linalg.cond(mat).mean()
        L = torch.linalg.eigvalsh(mat)
        return cond, L

    def forward(self, m_pred, m_true):
        """distance from m (predicted distance mat) to mt (true).
        Uses the (asymmetric) LogDet regularization D_{ld}(m_true,m_pred).
        The LogDet(M_pred*M_true^{-1}) term will be -inf if the prediction matrix is not full-rank
        (i.e. some pair of taxa have all-zero distances - SHOULD only happen if their input seqs are identical),
        so we add random*eps to the diagonal.
        Assumes that loss function already normalizes by # or nonzero values in m_pred.
        """

        if m_true.shape[-1] != m_true.shape[-2]:
            m_true, m_pred = (
                squareform(x, add_diagonal=not self.as_covariance).double()
                for x in (m_true, m_pred)
            )
            # m_pred = squareform(
            #     m_pred, add_diagonal=self.as_covariance)#.double()
        if self.radial:
            m_true, m_pred = ((-(x**2)).exp() for x in (m_true, m_pred))
        if m_true.dim() == 2:
            d = m_pred.shape[0]
            # if torch.matrix_rank(m) < d:
            #     return torch.tensor((float('inf')), dtype=torch.double)
            denom = m_pred.count_nonzero()
            with torch.autocast(enabled=False):
                m = m_true.double()
                m_pred = m_pred.double()
                mm_pred_inv = m.matmul(torch.pinverse(m_pred))
                # TODO: why this has negative determinant?
                lam = torch.diag(torch.rand(d)).to(m_pred.device) * self.eps
            # reg = mm_pred_inv.trace()-(mm_pred_inv+lam).logdet()-d
            sign, logabsdet = torch.slogdet(mm_pred_inv + lam)
            reg = mm_pred_inv.trace() - logabsdet * sign - d  # d or denom/2?
            return reg / denom
        else:

            b, n, _ = m_pred.shape
            size = b * n
            eps = torch.eye(n).repeat(b, 1, 1).to(m_true.device) * self.eps
            m_true = m_true + eps * m_true.norm()
            m_pred = m_true + eps * m_pred.norm()
            lam, V = torch.linalg.eigh(m_true)
            theta, U = torch.linalg.eigh(m_pred)
            lam = lam.relu()  # this is equiv to taking all nonzero eigs
            theta = theta.relu()
            # keep = min((lam > 0).sum(1).min(), (theta < 0).sum(1).min()).item()
            # if keep < n:
            #     lam, V = torch.lobpcg(m_true, k=keep)
            #     theta, U = torch.lobpcg(m_pred, k=keep)

            VtU = torch.bmm(V.transpose(1, 2), U) ** 2
            # first dim  # .sum(1).sum(1)
            mask = theta > 0
            theta_inv = torch.zeros_like(theta)
            theta_inv[mask] = 1 / theta[mask]

            X = torch.einsum("bi,bj->bij", lam, theta_inv)
            log_X = torch.zeros_like(X)

            mask = X > 0
            X_pos = X[mask]
            log_X[mask] = X_pos - X_pos.log() - torch.ones_like(X_pos)
            loss = torch.einsum("bij,bij->", VtU, log_X) / b
            # trace, summed along batches and normalized by # of taxa in all batches
            # reg = x.diagonal(offset=0, dim1=-1, dim2=-2).mean()
            # normalize by batch size
        if self.track_conditioning:
            lam_pos = lam.count_nonzero() / size
            theta_pos = lam.count_nonzero() / size

            return loss, lam_pos, theta_pos
        return loss


class VonNeumannLoss(torch.nn.Module):

    def __init__(self, format="distance", topk: int | None = None, reduce="mean"):
        """build regularized loss with base loss (default=MSE).
        If inverse is True, compute D_VN(X||Y)=tr(X log X - X logY - X +Y ).
        If reduce is 'mean', take the sum of all traces, normalize by n_batches and n_taxa
        """
        super().__init__()
        self.topk = topk
        self.as_covariance = format == "covariance"
        self.reduce = reduce

    def forward(self, m_pred: torch.Tensor, m_true: torch.Tensor):
        r"""distance from m (predicted distance mat) to mt (true).
        Uses the (asymmetric) von Neumann regularization.
        While LogDet requires rnk(M_pred) = rnk(M), VN only requires that rnk(M_pred) <= rnk(M).
        Best for covariance regression, otherwise we have to convert the
        (conditionally negative definite) distance matrix to a covariance matrix with  :math:`X\mapsto\exp(-\gamma X)`.
        uses eigendecomposition a la Kulis &al"""
        # TODO: use just sparse_coo_tensors, or eigendecomposition with only R top eigenvalues/vectors
        # TODO: guard against special case where batch_size==(n_taxa choose 2)
        if m_true.shape[-1] != m_true.shape[-2] or m_pred.shape[-1] != m_pred.shape[-2]:
            m_true = squareform(m_true, add_diagonal=not self.as_covariance).double()
            m_pred = squareform(m_pred, add_diagonal=not self.as_covariance).double()
        m_true_exp = torch.matrix_exp(-m_true)
        m_pred_exp = torch.matrix_exp(-m_pred)
        # pred from true
        d_vn = (m_true_exp @ (m_pred - m_true) - m_true_exp + m_pred_exp).diagonal(
            offset=0, dim1=-1, dim2=-2
        )
        if self.reduce == "mean":
            return d_vn.mean()
        elif self.reduce == "sum":
            return d_vn.sum()


# def LogDetLoss(eps=1e-5, eta=.5, loss=mse_loss, as_list=False, inverse=False):
#     '''build LogDet-regularized loss with base loss (default=MSE)'''
#     def log_det_loss(m, mt):
#         '''distance from m (predicted distance mat) to mt (true).
#         Uses the (asymmetric) LogDet regularization.
#         The LogDet(M_pred*M_true^{-1}) term will be -inf if the prediction matrix is not full-rank
#         (i.e. some pair of taxa have all-zero distances - SHOULD only happen if their input seqs are identical), so we add random*eps to the diagonal.
#         Assumes that loss function already normalizes by # or nonzero values in mt.'''
#         d = mt.shape[0]
#         # if torch.matrix_rank(m) < d:
#         #     return torch.tensor((float('inf')), dtype=torch.double)
#         denom = mt.count_nonzero()
#         if inverse:
#             m, mt = mt, m
#         with autocast(enabled=False):
#             m = m.double()
#             mt = mt.double()
#             mmt_inv = m.matmul(torch.pinverse(mt))
#             # TODO: why this has negative determinant?
#             lam = torch.diag(torch.rand(d)).to(mt.device) * eps
#         # reg = mmt_inv.trace()-(mmt_inv+lam).logdet()-d
#         sign, logabsdet = torch.slogdet(mmt_inv+lam)
#         reg = mmt_inv.trace()-logabsdet*sign-d  # d or denom/2?
#         return (eta*reg)/denom + loss(m, mt)

#     def cumulative_loss(m_pred, m_true):
#         return sum(log_det_loss(m, mt) for mt, m in zip(m_true, m_pred))/len(m_true)
#     if as_list:
#         return cumulative_loss
#     else:
#         return log_det_loss


vn_loss = VonNeumannLoss()
log_det_loss = LogDetLoss()


def logdet_loop(lam, V, theta, U):
    b, m, n = V.shape
    r = 0
    log_lam = lam.log()
    log_theta = theta.log()
    for k in range(b):
        for i in range(m):
            for j in range(n):
                s = (V[k, i, :] @ U[k, j, :]) ** 2 * (
                    lam[k, i] / theta[k, j] - log_lam[k, i] + log_theta[k, j] - 1
                )
                r += s

    return r / b


def vn_loop(lam, V, theta, U):
    b, m, n = V.shape
    reg = torch.zeros(b)
    r = 0
    log_lam = lam.log()
    log_theta = theta.log()
    for k in range(b):
        for i in range(m):
            for j in range(n):
                s = (V[k, i, :] @ U[k, j, :]) ** 2 * (
                    lam[k, i] * log_lam[k, i]
                    - lam[k, i] * log_theta[k, j]
                    - lam[k, j]
                    + theta[k, j]
                )
                r += s

    return r / b


# from multiprocess import pool


class L21Loss(torch.nn.Module):
    def __init__(self):
        """Matrix L_{2,1} norm.  More robust to outlier trees."""
        super().__init__()
        self.loss = torch.nn.MSELoss(reduction="none")

    def forward(self, ypred: torch.Tensor, y: torch.Tensor):
        d = self.loss(ypred, y)
        return d.mean(-1).sqrt().mean()  # mean over batches


class RelativeLoss(torch.nn.Module):
    def __init__(self, loss=torch.nn.L1Loss):
        """Loss normalized by y magnitude.  May be useful for BIONJ."""
        super().__init__()
        self.loss = loss(reduction="none")

    def forward(self, ypred: torch.Tensor, y: torch.Tensor):
        d = self.loss(ypred, y) / y
        return d.mean()  # mean over batches


class LSELoss(torch.nn.Module):

    def __init__(self, alpha: float = 1.0, ceil=False, l21=False):
        """build log-sum-exp loss.
        Goal is to achieve max|D_ij-D_ij^T| < min_ij D_ij^T / 2.
        If ceil is True, then we penalize only values > min_ij D_ij^T / 2."""
        super().__init__()
        self.alpha = alpha
        self.l21 = l21
        self.ceil = ceil
        self.loss = torch.nn.L1Loss(reduction="none")

    def forward(self, ypred: torch.Tensor, y: torch.Tensor):
        B, N = y.shape

        d = self.loss(ypred, y)
        d = (
            (self.alpha * (d)).logsumexp(-1)
            - torch.log(torch.Tensor([N]).to(ypred.device))
        ) / self.alpha  # scale by log(N) for logging purposes
        if self.ceil:
            d = (d - y.min(-1).values / 2.0).relu()
        if self.l21:
            d = d.sqrt()
        return d.mean()  # mean over batches


class SoftMaxLoss(torch.nn.Module):
    def __init__(self, alpha: float = 1.0):
        """build softmax loss.
        Goal is to achieve max|D_ij-D_ij^T| < min_ij D_ij^T / 2"""
        super().__init__()
        self.alpha = alpha
        self.loss = torch.nn.L1Loss(reduction="none")

    def forward(self, ypred: torch.Tensor, y: torch.Tensor):
        d = self.loss(ypred, y)
        d = (self.alpha * d).softmax(-1) * d
        return d.mean()  # mean over batches


# TODO for covariance matrices, don't need to exponentiate to make positive semidefinite
# if m_true.dim() == 2:
#     # TODO: this is NOT correct.  Use taylor expansion to find approximate *matrix logarithm*
#     x = {m_pred_log.nan_to_num(neginf=0) - m_true_log.nan_to_num(neginf=0)}
#     # some entries (e.g. diagonals) WILL be zero. torch treats inf-inf as nan
#     # log_pred_true = log_pred_true
#     # log_m_true = log_m_true.masked_fill(~torch.isfinite(log_m_true), 0)

#     x = m_pred * x - m_pred + m_true

#     denom = m_true.count_nonzero()
#     reg = torch.trace(x) / denom
# else:
#     b = len(m_true)
#     lam, V = torch.linalg.eigh(m_pred)
#     theta, U = torch.linalg.eigh(m_true)

#     if self.topk is not None:
#         ix = lam.argsort(descending=True)[:, : self.topk]
#         lam = lam.gather(1, ix)
#         V = V.gather(1, ix)  # todo: how to broadcast?

#     VtU = torch.bmm(V, U.transpose(1, 2)) ** 2
#     # first dim  # .sum(1).sum(1)
#     reg = (
#         torch.einsum("bi,bi,bij->", lam, lam.log(), VtU)
#         - torch.einsum("bij,bi->", VtU, lam)
#         + torch.einsum("bij,bj->", VtU, theta)
#         - torch.einsum("bi,bij,bj->", lam, VtU, theta.log())
#     )
# trace, summed along batches and normalized by # of taxa in all batches
# reg = x.diagonal(offset=0, dim1=-1, dim2=-2).mean()
# return reg / b
