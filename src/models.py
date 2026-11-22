"""Lightweight spatio-temporal model.

  Z = f_temporal(X)            temporal CNN shared by all sensors      (B, N, d)
  H = ReLU(A_hat Z W_g)        graph convolution across the node dim   (B, N, d)
  Y = f_head(H)                MLP -> 4 future steps                   (B, 4, N)

adj_mode:
  none     : temporal-only baseline, Y = f_head(Z)   (same encoder + head, no graph conv)
  fixed    : A_hat = normalised road-distance adjacency (buffer, never updated)
  adaptive : A = Softmax_row( ReLU(E1 E2^T) ),  E1, E2 in R^{N x d_e} learned
  hybrid   : A = alpha * A_fixed_row + (1 - alpha) * A_adaptive,  alpha = sigmoid(learned logit)
             (both row-stochastic -> consistently normalised, result is row-stochastic too)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Two 1-D convolutions over the 8 input steps (second one dilated), then a linear
    projection of the flattened (channels x time) map to d features. Shared across sensors."""

    def __init__(self, in_len=8, d=32, k=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, d, k, padding=k // 2)
        self.conv2 = nn.Conv1d(d, d, k, padding=2 * (k // 2), dilation=2)
        self.proj = nn.Linear(d * in_len, d)

    def forward(self, x):                      # x: (B, T_in, N)
        B, T, N = x.shape
        h = x.permute(0, 2, 1).reshape(B * N, 1, T)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.proj(h.flatten(1)))
        return h.view(B, N, -1)                # (B, N, d)


class GraphConv(nn.Module):
    """H = ReLU(A_hat Z W_g); the A_hat product mixes the NODE dimension independently per batch item."""

    def __init__(self, d):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)

    def forward(self, Z, A):                   # Z: (B, N, d), A: (N, N)
        return F.relu(self.W(torch.einsum("nm,bmd->bnd", A, Z)))


class Head(nn.Module):
    def __init__(self, d, out_len=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, out_len))

    def forward(self, H):                      # (B, N, d) -> (B, out_len, N)
        return self.net(H).permute(0, 2, 1)


class STGNN(nn.Module):
    def __init__(self, num_nodes, adj_mode="fixed", A_fixed=None, A_fixed_row=None, d=32, emb_dim=10,
                 topk=0, alpha=None, in_len=8, out_len=4):
        super().__init__()
        self.adj_mode, self.topk = adj_mode, int(topk)
        self.encoder = TemporalEncoder(in_len, d)
        self.head = Head(d, out_len)
        if adj_mode != "none":
            self.gconv = GraphConv(d)
        if adj_mode == "fixed":
            self.register_buffer("A_fixed", torch.as_tensor(A_fixed, dtype=torch.float32))
        if adj_mode in ("adaptive", "hybrid"):
            # random init as in Graph WaveNet; learned end-to-end by backprop
            self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
            self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        if adj_mode == "hybrid":
            self.register_buffer("A_fixed", torch.as_tensor(A_fixed_row, dtype=torch.float32))
            if alpha is None:   # learn alpha in (0,1), initialised at 0.5
                self.alpha_logit = nn.Parameter(torch.zeros(()))
            else:
                self.register_buffer("alpha_logit", torch.logit(torch.tensor(float(alpha))))

    # ------------------------------------------------------------------ adjacency
    def adaptive_adj(self):
        logits = F.relu(self.E1 @ self.E2.T)                      # (N, N), >= 0
        if self.topk > 0:
            # Sparsify BEFORE normalisation: keep the k largest scores per row, set the rest to
            # -inf so softmax assigns them exactly 0 and each row still sums to 1. Gradients flow
            # through the kept entries (and their E1/E2 rows); the top-k index choice itself is a
            # non-differentiable selection that is re-computed every forward pass.
            vals, idx = logits.topk(self.topk, dim=1)
            masked = torch.full_like(logits, float("-inf")).scatter(1, idx, vals)
            return torch.softmax(masked, dim=1)
        return torch.softmax(logits, dim=1)

    def alpha(self):
        return torch.sigmoid(self.alpha_logit) if self.adj_mode == "hybrid" else None

    def get_adj(self):
        if self.adj_mode == "fixed":
            return self.A_fixed
        if self.adj_mode == "adaptive":
            return self.adaptive_adj()
        if self.adj_mode == "hybrid":
            a = self.alpha()
            return a * self.A_fixed + (1 - a) * self.adaptive_adj()
        return None

    def forward(self, x):                      # x: (B, 8, N) normalised
        Z = self.encoder(x)
        if self.adj_mode == "none":
            return self.head(Z)
        H = self.gconv(Z, self.get_adj())
        return self.head(H)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
