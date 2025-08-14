#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmog_loss_tuning.py
---------------------------------
Loss-based (SBM-style) tuning to replace PCA in the clustering pipeline.

- Learns soft cluster assignments q (N × K), a block affinity matrix C (K × K),
  and a global bias b by maximizing an edge likelihood on a train split.
- Uses a held-out validation split to compute an ELBO-style objective (avg log-likelihood).
- Hyperparameters are tuned with Bayesian optimization (skopt).
- Produces hard labels via argmax over q after training with the best configuration.

This file is self-contained and does NOT use PCA.
"""

import os
import gc
import json
import math
import shutil
import random
import pickle
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scipy.sparse import load_npz, coo_matrix
from joblib import Parallel, delayed
from tqdm import tqdm

from skopt import Optimizer
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args


# ---------------------------- Utils ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_empty_cache(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def estimate_max_jobs_per_gpu(required_per_job_mb=4000):
    """
    Estimate the maximum number of jobs that can run concurrently on the available GPU memory
    assuming each job requires ~4000MB by default
    """
    if not torch.cuda.is_available():
        return 1  # CPU fallback

    total_free = 0
    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.mem_get_info(i)
        free_mb = stats[0] / (1024 ** 2)  # bytes → MB
        total_free += free_mb

    max_jobs = max(int(total_free // required_per_job_mb), 1)
    return max_jobs


# ---------------------------- Data split ----------------------------

def train_val_split(adj: coo_matrix, val_frac: float = 0.2, seed: int = 42):
    """Return index split and corresponding subgraphs (COO)."""
    set_seed(seed)
    N = adj.shape[0]
    perm = np.random.permutation(N)
    n_val = int(N * val_frac)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    train_adj = adj.tocsr()[train_idx][:, train_idx].tocoo()
    val_adj = adj.tocsr()[val_idx][:, val_idx].tocoo()
    return train_idx, val_idx, train_adj, val_adj


# ---------------------------- SBM-style model ----------------------------

@dataclass
class SbmConfig:
    K: int
    lr: float = 1e-2
    n_epochs: int = 500
    batch_edges: int = 4096
    neg_ratio: float = 1.0         # negatives per positive edge
    optimizer: str = "Adam"        # 'Adam' | 'AdamW' | 'SGD'
    val_steps: int = 30            # steps to optimize q on validation (E-step on val)
    device: str = "cpu"
    seed: int = 42


class SbmModel(nn.Module):
    """
    Simple symmetric SBM with logistic link:
        p(A_ij=1) = sigmoid( q_i^T C q_j + b )
    - q_logits: (N, K) free logits, q = softmax(q_logits)
    - C: (K, K) block affinity (symmetric)
    - b: scalar bias
    """
    def __init__(self, N: int, config: SbmConfig):
        super().__init__()
        self.N = N
        self.config = config

        self.q_logits = nn.Parameter(torch.zeros(N, config.K))
        # symmetric C initialization
        C = 0.01 * torch.randn(config.K, config.K)
        C = 0.5 * (C + C.t())
        self.C = nn.Parameter(C)
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _bce_log_prob(logit, y):
        # log p = y*logσ + (1-y)*log(1-σ) = -BCE
        return -nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")

    def forward_logits(self, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        # logits for edges (i,j)
        q = nn.functional.softmax(self.q_logits, dim=-1)  # (N, K)
        qi = q[i_idx]   # (B, K)
        qj = q[j_idx]   # (B, K)
        # bilinear form qi^T C qj
        z = (qi @ self.C) * qj
        z = z.sum(dim=-1) + self.bias  # (B,)
        return z

    def elbo_batch(self, i_idx, j_idx, y):
        logits = self.forward_logits(i_idx, j_idx)
        return self._bce_log_prob(logits, y).mean()

    @torch.no_grad()
    def hard_labels(self) -> np.ndarray:
        q = nn.functional.softmax(self.q_logits, dim=-1)  # (N,K)
        return torch.argmax(q, dim=-1).cpu().numpy()


# ---------------------------- Sampling ----------------------------

def sample_edges(adj: coo_matrix, num_samples: int, rng: np.random.Generator):
    """Sample positive edges (i,j) uniformly from non-zeros (upper triangle if symmetric)."""
    # Ensure COO
    adj = adj.tocoo()
    rows, cols = adj.row, adj.col

    # If graph is symmetric, sample only i<j
    mask = rows < cols
    pos_r = rows[mask]
    pos_c = cols[mask]
    M = len(pos_r)
    if M == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    idx = rng.integers(0, M, size=min(num_samples, M), endpoint=False)
    return pos_r[idx], pos_c[idx]


def sample_negatives(N: int, num_samples: int, rng: np.random.Generator):
    """Uniformly sample node pairs as negatives. In sparse graphs, collision with real edges is rare."""
    i = rng.integers(0, N, size=num_samples, endpoint=False)
    j = rng.integers(0, N, size=num_samples, endpoint=False)
    # Avoid i==j
    same = (i == j)
    if same.any():
        j[same] = (j[same] + 1) % N
    return i, j


# ---------------------------- Training & Validation ----------------------------

def train_sbm(adj: coo_matrix, config: SbmConfig) -> Tuple[SbmModel, float, float]:
    """
    Train SBM on 'adj' with negative sampling.
    Returns: (model, initial_train_elbo, final_train_elbo)
    """
    device = config.device
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    N = adj.shape[0]
    model = SbmModel(N, config).to(device)

    if config.optimizer == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "AdamW":
        opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=config.lr)

    # Precompute positives
    pos_r_all, pos_c_all = sample_edges(adj, adj.nnz, rng)  # may be large; we'll subsample inside loop
    initial_elbo, final_elbo = None, None

    for epoch in tqdm(range(config.n_epochs), desc="SBM-Train"):
        # Mini-batch: subsample positives, then negatives
        pos_r, pos_c = sample_edges(adj, config.batch_edges, rng)
        if pos_r.size == 0:
            break
        neg_num = int(config.batch_edges * config.neg_ratio)
        neg_r, neg_c = sample_negatives(N, neg_num, rng)

        # Build tensors
        i_idx = torch.from_numpy(np.concatenate([pos_r, neg_r])).long().to(device)
        j_idx = torch.from_numpy(np.concatenate([pos_c, neg_c])).long().to(device)
        y = torch.cat([torch.ones(len(pos_r)), torch.zeros(len(neg_r))]).float().to(device)

        opt.zero_grad(set_to_none=True)
        loss = -model.elbo_batch(i_idx, j_idx, y)  # maximize ELBO -> minimize -ELBO
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if epoch == 0:
            initial_elbo = -loss.item()

        if (epoch + 1) % 50 == 0:
            final_elbo = -loss.item()

    if final_elbo is None:
        final_elbo = initial_elbo if initial_elbo is not None else 0.0

    gc.collect()
    safe_empty_cache(device)
    return model, float(initial_elbo), float(final_elbo)


@torch.no_grad()
def val_elbo(adj_val: coo_matrix, model: SbmModel, steps: int = 30) -> float:
    """
    Compute validation ELBO with a local E-step on q for validation nodes,
    keeping C and bias fixed.
    """
    device = next(model.parameters()).device
    N_val = adj_val.shape[0]
    K = model.config.K

    # Clone a local copy of q_logits for val nodes
    q_logits_val = nn.Parameter(torch.zeros(N_val, K, device=device))
    opt = torch.optim.SGD([q_logits_val], lr=0.1)

    # Prepare positive and negative pairs
    rng = np.random.default_rng(12345)
    pos_r, pos_c = sample_edges(adj_val, min(adj_val.nnz, 50000), rng)
    if pos_r.size == 0:
        return -1e9  # no edges -> very bad
    neg_r, neg_c = sample_negatives(N_val, len(pos_r), rng)

    i_idx = torch.from_numpy(np.concatenate([pos_r, neg_r])).long().to(device)
    j_idx = torch.from_numpy(np.concatenate([pos_c, neg_c])).long().to(device)
    y = torch.cat([torch.ones(len(pos_r)), torch.zeros(len(neg_r))]).float().to(device)

    # Temporary module that reuses model.C and bias but with local q
    class _ValWrap(nn.Module):
        def __init__(self, C, bias, q_logits_ref):
            super().__init__()
            self.C = C
            self.bias = bias
            self.q_logits_ref = q_logits_ref
        def forward_logits(self, i_idx, j_idx):
            q = nn.functional.softmax(self.q_logits_ref, dim=-1)
            qi = q[i_idx]; qj = q[j_idx]
            z = (qi @ self.C) * qj
            return z.sum(dim=-1) + self.bias

    wrap = _ValWrap(model.C, model.bias, q_logits_val)

    # Optimize q on validation nodes
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = wrap.forward_logits(i_idx, j_idx)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        opt.step()

    # Compute average log-likelihood (ELBO approximation)
    logits = wrap.forward_logits(i_idx, j_idx)
    ll = -nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    return float(ll.item())


# ---------------------------- Hyperparameter search ----------------------------

# Search space (you can widen as needed)
space = [
    Integer(64, 2048, name='k'),               # number of clusters
    Real(1e-4, 5e-2, prior='log-uniform', name='learning_rate'),
    Integer(200, 1500, name='n_epochs'),
    Integer(1024, 16384, name='batch_edges'),
    Real(0.5, 2.0, name='neg_ratio'),
    Categorical(['Adam', 'AdamW', 'SGD'], name='optimizer'),
]


@use_named_args(space)
def objective(**params):
    """Minimize negative validation ELBO (maximize ELBO)."""
    # Load graph
    file_path = "./sparse_connectivity_matrix.npz"
    adj = load_npz(file_path).tocsr()
    N = adj.shape[0]
    # ensure symmetry & make COO
    adj = (adj + adj.T).multiply(0.5).tocsr()
    adj_coo = adj.tocoo()

    # Split
    train_idx, val_idx, train_adj, val_adj = train_val_split(adj_coo, val_frac=0.2, seed=42)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(0)

    # Config
    cfg = SbmConfig(
        K=int(params["k"]),
        lr=float(params["learning_rate"]),
        n_epochs=int(params["n_epochs"]),
        batch_edges=int(params["batch_edges"]),
        neg_ratio=float(params["neg_ratio"]),
        optimizer=str(params["optimizer"]),
        val_steps=30,
        device=device,
        seed=42,
    )

    # Train on train graph
    model, init_elbo, final_elbo = train_sbm(train_adj, cfg)

    # Validate
    vll = val_elbo(val_adj, model, steps=cfg.val_steps)
    print(f"[VAL] ELBO={vll:.6f}  (init={init_elbo:.6f}, final={final_elbo:.6f})")

    # Cleanup
    del model, adj, adj_coo, train_adj, val_adj
    gc.collect(); safe_empty_cache(device)

    return -vll  # skopt minimizes


def main():
    set_seed(42)
    total_trials = 20  # adjust as needed

    # Parallel policy: 1 per GPU, else 1 job
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_jobs_per_batch = max(1, gpu_count)
    n_batches = (total_trials + n_jobs_per_batch - 1) // n_jobs_per_batch

    opt = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI", random_state=42)
    space_names = [d.name for d in space]

    best_loss = float("inf")
    best_cfg = None

    for bi in range(n_batches):
        candidates = opt.ask(n_points=n_jobs_per_batch)
        if not isinstance(candidates, list):
            candidates = [candidates]

        def run_one(point):
            params = {name: val for name, val in zip(space_names, point)}
            return objective(**params)

        scores = Parallel(n_jobs=n_jobs_per_batch)(delayed(run_one)(pt) for pt in candidates)
        opt.tell(candidates, scores)

        cur_best = min(opt.yi)
        if cur_best < best_loss:
            best_loss = cur_best
            best_point = opt.Xi[int(np.argmin(opt.yi))]
            best_cfg = {name: val for name, val in zip(space_names, best_point)}

        print(f"[Batch {bi+1}/{n_batches}] best ELBO so far = {-best_loss:.6f}")

    print("\nBest configuration:")
    print(best_cfg)
    print(f"Best validation ELBO: {-best_loss:.6f}")

    # --- Train final model on FULL graph with best config and save labels ---
    file_path = "./sparse_connectivity_matrix.npz"
    adj_full = load_npz(file_path).tocsr()
    adj_full = (adj_full + adj_full.T).multiply(0.5).tocsr().tocoo()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(0)

    cfg = SbmConfig(
        K=int(best_cfg["k"]),
        lr=float(best_cfg["learning_rate"]),
        n_epochs=int(best_cfg["n_epochs"]),
        batch_edges=int(best_cfg["batch_edges"]),
        neg_ratio=float(best_cfg["neg_ratio"]),
        optimizer=str(best_cfg["optimizer"]),
        val_steps=30,
        device=device,
        seed=42,
    )

    model, init_elbo, final_elbo = train_sbm(adj_full, cfg)
    labels = model.hard_labels()

    os.makedirs("vmog_runs", exist_ok=True)
    np.save("vmog_runs/sbm_labels_FINAL.npy", labels)
    torch.save({
        "q_logits": model.q_logits.detach().cpu(),
        "C": model.C.detach().cpu(),
        "bias": model.bias.detach().cpu(),
        "config": cfg.__dict__,
        "best_cfg": best_cfg,
        "best_val_elbo": float(-best_loss),
    }, "vmog_runs/sbm_model_FINAL.pt")
    print("[INFO] Saved labels to vmog_runs/sbm_labels_FINAL.npy and model to vmog_runs/sbm_model_FINAL.pt")


if __name__ == "__main__":
    main()
