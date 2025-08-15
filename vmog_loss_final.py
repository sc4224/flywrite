<<<<<<< HEAD

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from tqdm import tqdm
from joblib import Parallel, delayed
from pathlib import Path
import gc
import json

# ------------------------------------
# Utilities
# ------------------------------------

def load_mapping(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    # mapping should map row index -> root_id
    return mapping

def sigmoid(x, clamp=False):
    if clamp:
        x = torch.clamp(x, min=-5, max=5)
    return 1.0 / (1.0 + torch.exp(-x))

# ------------------------------------
# Mini-batch GMM with sbm-style M-step
# ------------------------------------

class MiniBatchGMM(nn.Module):
    """GMM trained via generalized EM in a streaming fashion:
      - E-step: compute responsibilities for the current mini-batch
      - M-step: gradient ascent on expected complete-data log-likelihood
                using only the current mini-batch (sbm.py style)
    Diagonal covariance; parameters are updated with an optimizer.
    """
    def __init__(self, n_components, n_features, device='cpu', dtype=torch.float32):
        super().__init__()
        self.K = n_components
        self.D = n_features
        self.device = device
        self.dtype = dtype

        # Parameters to learn (initialized later)
        self.means = nn.Parameter(torch.zeros(self.K, self.D, dtype=dtype, device=device))
        # log-variance per component & feature (diagonal covariance)
        self.log_vars = nn.Parameter(torch.zeros(self.K, self.D, dtype=dtype, device=device))
        # unnormalized logits for mixture weights
        self.logits = nn.Parameter(torch.zeros(self.K, dtype=dtype, device=device))

        # buffers for results
        self.labels_ = None

    def initialize_params_from_sample(self, X_sample):
        """X_sample: (B, D) torch tensor on device"""
        B = X_sample.size(0)
        # random subset for means init
        idx = torch.randperm(B, device=X_sample.device)[:min(self.K, B)]
        means_init = X_sample[idx]
        if means_init.size(0) < self.K:
            # pad by repeating
            pad = self.K - means_init.size(0)
            means_init = torch.cat([means_init, means_init[:pad]], dim=0)
        with torch.no_grad():
            self.means.copy_(means_init)
            var = X_sample.var(dim=0, unbiased=False) + 1e-3
            self.log_vars.copy_(var.expand(self.K, -1).log_())
            self.logits.zero_()

    def e_step_batch(self, Xb):
        """Compute responsibilities for a batch.
        Xb: (B, D) on device -> returns: resp (B, K)"""
        K, D = self.K, self.D
        B = Xb.size(0)
        log_resp = torch.empty(B, K, dtype=self.dtype, device=self.device)
        log_pi = F.log_softmax(self.logits, dim=0)  # (K,)
        inv_vars = torch.exp(-self.log_vars)        # (K, D)
        const = -0.5 * D * np.log(2 * np.pi)
        for k in range(K):
            diff = Xb - self.means[k].unsqueeze(0)           # (B, D)
            term1 = const - 0.5 * torch.sum(self.log_vars[k])  # scalar
            term2 = -0.5 * torch.sum(diff * diff * inv_vars[k].unsqueeze(0), dim=1)  # (B,)
            log_prob = term1 + term2                           # (B,)
            log_resp[:, k] = log_pi[k] + log_prob
        log_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
        resp = torch.exp(log_resp - log_norm)  # (B, K)
        return resp

    def expected_complete_loglik_batch(self, Xb, resp):
        """Compute expected complete-data log-likelihood for batch (sum over batch)."""
        K, D = self.K, self.D
        log_pi = F.log_softmax(self.logits, dim=0)  # (K,)
        inv_vars = torch.exp(-self.log_vars)        # (K, D)
        elbo = torch.zeros((), dtype=self.dtype, device=self.device)
        const = -0.5 * D * np.log(2 * np.pi)
        for k in range(K):
            diff = Xb - self.means[k].unsqueeze(0)  # (B, D)
            quad = -0.5 * torch.sum(diff * diff * inv_vars[k].unsqueeze(0), dim=1)  # (B,)
            log_det = -0.5 * torch.sum(self.log_vars[k])  # scalar
            log_comp = log_pi[k] + const + log_det + quad  # (B,)
            elbo += torch.sum(resp[:, k] * log_comp)  # scalar
        return elbo

    def fit_streaming(self,
                      csr_X: csr_matrix,
                      batch_size=64,
                      n_epochs=3,
                      m_steps_per_batch=1,
                      lr=1e-2,
                      verbose=True):
        """Train with streaming mini-batches from a CSR sparse matrix (rows are samples)."""
        N, D = csr_X.shape
        assert D == self.D

        # init
        init_idx = np.random.default_rng(42).choice(N, size=min(512, N), replace=False)
        X_init = torch.tensor(csr_X[init_idx].toarray(), dtype=self.dtype, device=self.device)
        self.initialize_params_from_sample(X_init)
        del X_init; gc.collect()

        opt = torch.optim.Adam([self.means, self.log_vars, self.logits], lr=lr)

        for epoch in range(n_epochs):
            perm = np.random.permutation(N)
            for t in tqdm(range(0, N, batch_size), desc=f"Epoch {epoch+1}/{n_epochs}"):
                idx = perm[t: t+batch_size]
                Xb_np = csr_X[idx].toarray().astype(np.float32)
                Xb = torch.tensor(Xb_np, dtype=self.dtype, device=self.device)

                # E-step for batch
                with torch.no_grad():
                    resp = self.e_step_batch(Xb)  # (B, K)

                # M-step (sbm-style): gradient ascent on expected complete-data loglik
                for _ in range(m_steps_per_batch):
                    opt.zero_grad(set_to_none=True)
                    elbo = self.expected_complete_loglik_batch(Xb, resp)
                    loss = -elbo
                    loss.backward()
                    opt.step()

                # free batch
                del Xb, Xb_np, resp, loss, elbo
                gc.collect()
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return self

    def predict_streaming(self, csr_X: csr_matrix, batch_size=1024):
        """Assign labels in a streaming way; returns (N,) long tensor on CPU."""
        N, D = csr_X.shape
        labels = []
        for t in tqdm(range(0, N, batch_size), desc="Assigning clusters"):
            idx = slice(t, min(t+batch_size, N))
            Xb_np = csr_X[idx].toarray().astype(np.float32)
            Xb = torch.tensor(Xb_np, dtype=self.dtype, device=self.device)
            with torch.no_grad():
                resp = self.e_step_batch(Xb)
                lab = torch.argmax(resp, dim=1).cpu()
            labels.append(lab)
            del Xb, Xb_np, resp, lab
            gc.collect()
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.labels_ = torch.cat(labels, dim=0)
        return self.labels_


# ------------------------------------
# PCA-free run function (streaming)
# ------------------------------------

def run_pca_gmm(batch, index, device="cuda",
                K=967,   # n_components
                epochs=3,
                batch_size=64,
                m_steps_per_batch=1,
                lr=1e-2):
    """PCA-free, streaming version using sbm.py-style M-step (gradient ascent on ELBO)."""
    torch_dtype = torch.float32 if device != 'cuda' else torch.bfloat16

    iteration = (batch * 4) + index

    # load CSR sparse features
    csr_X = load_npz("./sparse_connectivity_matrix.npz").tocsr()
    N, D = csr_X.shape
    print(f"[run_pca_gmm-noPCA/stream] CSR shape={N}x{D}, nnz={csr_X.nnz}")

    model = MiniBatchGMM(n_components=K, n_features=D, device=device, dtype=torch_dtype)

    model.fit_streaming(csr_X, batch_size=batch_size, n_epochs=epochs,
                        m_steps_per_batch=m_steps_per_batch, lr=lr, verbose=True)

    labels = model.predict_streaming(csr_X, batch_size=1024)   # (N,)

    Path("vmog_runs").mkdir(parents=True, exist_ok=True)
    torch.save(model.means.detach().cpu().float(), f"vmog_runs/vmog_cluster_centers_tuned_{iteration}.pt")
    torch.save(labels.cpu(), f"vmog_runs/vmog_labels_tuned_{iteration}.pt")

    # optional mapping save
    try:
        with open('./root_id_to_index_mapping.json','r',encoding='utf-8') as f:
            mapping = json.load(f)
        if isinstance(mapping, dict):
            cluster_assignment_dict = { mapping.get(str(i), i): int(labels[i].item()) for i in range(len(labels)) }
        else:
            cluster_assignment_dict = { mapping[i]: int(labels[i].item()) for i in range(len(labels)) }
        import numpy as _np
        _np.save(f"vmog_runs/vmog_cluster_assignment_dict_tuned_{iteration}.npy", cluster_assignment_dict, allow_pickle=True)
    except Exception as e:
        print("[warn] mapping not saved:", e)

    print("Done.")
    return labels


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 4
    n_batches = 12
    for batch in range(n_batches):
        _ = Parallel(n_jobs=batch_size)(
            delayed(run_pca_gmm)(batch, index=i+1, device=device)
            for i in range(batch_size)
        )
=======
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vmog_pca_final.py (NO-PCA EDITION)
----------------------------------
This version removes PCA entirely and replaces the embedding+GMM stage with a
loss-based SBM-style clustering directly on the graph. Apart from removing PCA,
the rest of the pipeline (saving labels, picking best run by a score, alignment,
plots, credible intervals, etc.) is kept as close as possible to the original.

Fill your desired hyperparameters in `best_params`.
"""

import os
import gc
import json
import shutil
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scipy.sparse import load_npz, coo_matrix
from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import SpectralEmbedding
from scipy.optimize import linear_sum_assignment

from joblib import Parallel, delayed
from tqdm import tqdm

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from index_mapping import load_mapping


# ========================= User-editable hyperparameters =========================
# Replace values as you wish. Keys are for the SBM-style model now (not PCA/GMM).
best_params = {
    'k': 974,                 # number of clusters
    'learning_rate': 1e-2,
    'n_epochs': 500,
    'batch_edges': 4096,
    'neg_ratio': 1.0,
    'optimizer': 'Adam',      # 'Adam' | 'AdamW' | 'SGD'
}

# ========================= Ground-truth (unchanged) =========================
with open("root_id_type_dict.pkl", "rb") as f:
    root_id_type_dict = pickle.load(f)
uniq_types = sorted(set(root_id_type_dict.values()))
cluster_string_to_cid = {t: idx for idx, t in enumerate(uniq_types)}


# ========================= Utilities (mostly unchanged) =========================

def safe_empty_cache(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def estimate_max_jobs_per_gpu(required_per_job_mb=4000):
    if not torch.cuda.is_available():
        return 1
    total_free = 0
    for i in range(torch.cuda.device_count()):
        free_mb = torch.cuda.mem_get_info(i)[0] / (1024 ** 2)
        total_free += free_mb
    max_jobs = max(int(total_free // required_per_job_mb), 1)
    return max_jobs


def compare_two_assignments(assignment_A, assignment_B):
    confusion_matrix_value = confusion_matrix(assignment_A, assignment_B)
    row_ind, col_ind = linear_sum_assignment(confusion_matrix_value, maximize=True)
    score = confusion_matrix_value[row_ind, col_ind].sum()
    aligned_confusion_matrix = confusion_matrix_value[:, col_ind]
    return score, aligned_confusion_matrix


def draw_graph(xy, edges, labels, title, filename=None):
    G = nx.Graph()
    pos = {i: (x, y) for i, (x, y) in enumerate(xy)}
    G.add_edges_from(edges)

    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(G, pos, node_color=labels, cmap="tab20", node_size=20, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=0.2, alpha=0.3)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)
    plt.close()


# ========================= SBM-style model (NEW) =========================

class SbmModel(nn.Module):
    """
    Simple symmetric SBM with logistic link:
        p(A_ij=1) = sigmoid( q_i^T C q_j + b )
    - q_logits: (N, K) free logits, q = softmax(q_logits)
    - C: (K, K) block affinity (symmetric)
    - b: scalar bias
    """
    def __init__(self, N: int, K: int):
        super().__init__()
        self.N = N
        self.K = K
        self.q_logits = nn.Parameter(torch.zeros(N, K))
        C = 0.01 * torch.randn(K, K)
        C = 0.5 * (C + C.t())
        self.C = nn.Parameter(C)
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _bce_log_prob(logit, y):
        # log p = y*logσ + (1-y)*log(1-σ) = -BCE
        return -nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")

    def forward_logits(self, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        q = nn.functional.softmax(self.q_logits, dim=-1)  # (N, K)
        qi = q[i_idx]   # (B, K)
        qj = q[j_idx]   # (B, K)
        z = (qi @ self.C) * qj
        z = z.sum(dim=-1) + self.bias  # (B,)
        return z

    def elbo_batch(self, i_idx, j_idx, y):
        logits = self.forward_logits(i_idx, j_idx)
        return self._bce_log_prob(logits, y).mean()

    @torch.no_grad()
    def hard_labels(self) -> torch.Tensor:
        q = nn.functional.softmax(self.q_logits, dim=-1)  # (N,K)
        return torch.argmax(q, dim=-1)  # tensor on current device


def sample_edges(adj: coo_matrix, num_samples: int, rng: np.random.Generator):
    """Sample positive edges (i,j) uniformly from non-zeros with i<j if symmetric."""
    adj = adj.tocoo()
    rows, cols = adj.row, adj.col
    mask = rows < cols
    pos_r = rows[mask]; pos_c = cols[mask]
    M = len(pos_r)
    if M == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    idx = rng.integers(0, M, size=min(num_samples, M), endpoint=False)
    return pos_r[idx], pos_c[idx]


def sample_negatives(N: int, num_samples: int, rng: np.random.Generator):
    i = rng.integers(0, N, size=num_samples, endpoint=False)
    j = rng.integers(0, N, size=num_samples, endpoint=False)
    same = (i == j)
    if same.any():
        j[same] = (j[same] + 1) % N
    return i, j


def train_sbm_on_full_graph(adj_csr, K: int, lr: float, n_epochs: int, batch_edges: int,
                            neg_ratio: float, optimizer: str, device: str, seed: int = 42) -> SbmModel:
    rng = np.random.default_rng(seed)
    N = adj_csr.shape[0]
    adj = (adj_csr + adj_csr.T).multiply(0.5).tocsr().tocoo()

    model = SbmModel(N, K).to(device)
    if optimizer == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "AdamW":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in tqdm(range(n_epochs), desc="SBM-Train"):
        pos_r, pos_c = sample_edges(adj, batch_edges, rng)
        if pos_r.size == 0:
            break
        neg_num = int(batch_edges * neg_ratio)
        neg_r, neg_c = sample_negatives(N, neg_num, rng)

        i_idx = torch.from_numpy(np.concatenate([pos_r, neg_r])).long().to(device)
        j_idx = torch.from_numpy(np.concatenate([pos_c, neg_c])).long().to(device)
        y = torch.cat([torch.ones(len(pos_r)), torch.zeros(len(neg_r))]).float().to(device)

        opt.zero_grad(set_to_none=True)
        loss = -model.elbo_batch(i_idx, j_idx, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        if (epoch + 1) % 100 == 0:
            print(f"[SBM] epoch {epoch+1}: -ELBO={loss.item():.6f}")

    gc.collect()
    safe_empty_cache(device)
    return model


def stratified_sample_by_labels(labels_np, max_per_label=50, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    idxs = []
    for lab in np.unique(labels_np):
        cand = np.where(labels_np == lab)[0]
        if cand.size == 0:
            continue
        take = cand if cand.size <= max_per_label else rng.choice(cand, size=max_per_label, replace=False)
        idxs.append(take)
    return np.concatenate(idxs) if len(idxs) else np.array([], dtype=int)


# ========================= Main per-run function (renamed logic, same name) =========================

def run_pca_gmm(batch, index, best_params, device="cuda"):
    """
    Keep the original entry name but now run SBM-style clustering on the full graph.
    Saves files under the same names expected later by the pipeline:
      - vmog_labels_tuned_{iteration}.pt : tensor of labels (N,)
      - vmog_cluster_centers_tuned_{iteration}.pt : here we save SBM's C matrix (K,K) for compatibility
      - vmog_cluster_assignment_dict_tuned_{iteration}.npy : mapping root_id -> label
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device)

    iteration_tag = f"{batch}_{index}"
    # original iteration formula kept
    n_jobs_per_batch = 1  # this will be overwritten in __main__ loop; keep placeholder
    iteration = batch * max(n_jobs_per_batch, 1) + index

    # --- Hyperparams from best_params ---
    K = int(best_params["k"])
    lr = float(best_params["learning_rate"])
    n_epochs = int(best_params["n_epochs"])
    batch_edges = int(best_params["batch_edges"])
    neg_ratio = float(best_params["neg_ratio"])
    optimizer = str(best_params["optimizer"])

    # --- Load graph ---
    file_path = "./sparse_connectivity_matrix.npz"
    adj_matrix = load_npz(file_path).tocsr()
    print(f"Loaded sparse matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

    # --- Train SBM on full graph ---
    model = train_sbm_on_full_graph(
        adj_csr=adj_matrix,
        K=K,
        lr=lr,
        n_epochs=n_epochs,
        batch_edges=batch_edges,
        neg_ratio=neg_ratio,
        optimizer=optimizer,
        device=device
    )

    # --- Labels & diagnostic ---
    labels = model.hard_labels()  # tensor on device
    uniq, cnt = torch.unique(labels, return_counts=True)
    print(f"[Diag] #clusters={len(uniq)}, top-5 sizes={cnt.topk(k=min(5, len(cnt))).values.tolist()}")

    # ---------- Silhouette (on stratified sample) ----------
    labels_np_all = labels.cpu().numpy()
    sample_indices = stratified_sample_by_labels(labels_np_all, max_per_label=50)
    if sample_indices.size == 0:
        print("Not enough clusters in sample, fallback-scoring this run")
        fallback_sil = 0.0
        os.makedirs("vmog_runs", exist_ok=True)
        with open("vmog_runs/gmm_scores.csv", "a") as f:
            f.write(f"{iteration},{fallback_sil:.4f}\n")
        return 1.0

    adj_sample = adj_matrix[sample_indices]
    labels_sample = labels_np_all[sample_indices]

    non_zero_mask = adj_sample.getnnz(axis=1) > 0
    adj_sample = adj_sample[non_zero_mask]
    labels_sample = labels_sample[non_zero_mask]

    if len(np.unique(labels_sample)) < 2:
        print("Not enough clusters in sample, fallback-scoring this run")
        fallback_sil = 0.0
        os.makedirs("vmog_runs", exist_ok=True)
        with open("vmog_runs/gmm_scores.csv", "a") as f:
            f.write(f"{iteration},{fallback_sil:.4f}\n")
        return 1.0

    score = silhouette_score(
        adj_sample,
        labels_sample,
        metric="cosine",   # sparse-friendly
        n_jobs=1
    )

    os.makedirs("vmog_runs", exist_ok=True)
    with open("vmog_runs/gmm_scores.csv", "a") as f:
        f.write(f"{iteration},{score:.4f}\n")
    print(f"[{iteration_tag}] silhouette = {score:.4f}")

    # --- Save results under original filenames ---
    # mapping root_id -> label
    mapping = load_mapping('./root_id_to_index_mapping.json')
    cluster_assignment_dict = {mapping[i]: int(labels.cpu().numpy()[i]) for i in range(len(labels))}

    torch.save(labels.cpu(), f"vmog_runs/vmog_labels_tuned_{iteration}.pt")
    # For compatibility: save SBM block matrix C as "cluster centers"
    torch.save(model.C.detach().cpu(), f"vmog_runs/vmog_cluster_centers_tuned_{iteration}.pt")
    np.save(f"vmog_runs/vmog_cluster_assignment_dict_tuned_{iteration}.npy", cluster_assignment_dict, allow_pickle=True)
    print("Saved SBM params and labels to disk.")

    # Cleanup
    del model, labels, adj_matrix, mapping, cluster_assignment_dict
    gc.collect()
    safe_empty_cache(device)

    return -score


def align_labels(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize matching
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_pred = np.array([mapping[label] for label in pred_labels])
    return aligned_pred


# ========================= Main =========================

if __name__ == "__main__":
    # Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    # Parallel scheduling (kept similar)
    total_runs = 1
    required_per_job_mb = 4000
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count <= 1:
        n_jobs_per_batch = 1
    else:
        n_jobs_per_batch = min(estimate_max_jobs_per_gpu(required_per_job_mb), gpu_count)
    n_batches = (total_runs + n_jobs_per_batch - 1) // n_jobs_per_batch

    print("\n[INFO] Running final clustering with loss-based tuning (no PCA)...")
    batch = 0
    index = 0

    # First run (single)
    run_pca_gmm(batch, index, best_params, device=device)

    # Additional runs for CIs (kept structure)
    for batch in range(n_batches):
        n_gpu = torch.cuda.device_count()
        _ = Parallel(n_jobs=n_jobs_per_batch)(
            delayed(run_pca_gmm)(
                batch,
                idx + 1,
                best_params,
                device=f"cuda:{idx % n_gpu}" if n_gpu else "cpu"
            )
            for idx in range(n_jobs_per_batch)
        )

    # --- Read scores & pick best iteration ---
    scores = {}
    csv_path = "vmog_runs/gmm_scores.csv"
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for line in f:
                tag, sil = line.strip().split(",")
                try:
                    scores[int(tag)] = float(sil)
                except ValueError:
                    continue
    else:
        print(f"[INFO] {csv_path} not found (first run?) — skip reading old scores")

    best_iter = max(scores, key=scores.get)
    best_score = scores[best_iter]
    print(f"Best silhouette score: {best_score:.4f} (iteration {best_iter})")

    # --- Save FINAL files (copy, avoids torch.load warnings) ---
    os.makedirs("vmog_runs", exist_ok=True)
    shutil.copyfile(f"vmog_runs/vmog_labels_tuned_{best_iter}.pt", "vmog_runs/vmog_labels_FINAL.pt")
    shutil.copyfile(f"vmog_runs/vmog_cluster_centers_tuned_{best_iter}.pt", "vmog_runs/vmog_cluster_centers_FINAL.pt")

    # --- Load final labels ---
    gmm_labels = torch.load("vmog_runs/vmog_labels_FINAL.pt", weights_only=True).cpu()
    uniq, cnt = torch.unique(gmm_labels, return_counts=True)
    print(f"[Diag Final] #clusters={len(uniq)}, top-5 sizes={cnt.topk(k=min(5, len(cnt))).values.tolist()}")
    gmm_labels = gmm_labels.numpy()

    # index_to_root_id
    with open("root_id_to_index_mapping.json", "r") as f:
        id_to_index = json.load(f)
    index_to_root_id = {int(v): int(k) for k, v in id_to_index.items()}

    # build filtered label lists
    gmm_root_ids = [index_to_root_id[i] for i in range(len(gmm_labels))]
    gt_labels = [root_id_type_dict.get(rid, -1) for rid in gmm_root_ids]
    valid_mask = np.array([gt != -1 for gt in gt_labels], dtype=bool)

    gmm_labels_filtered = np.array(gmm_labels)[valid_mask]
    gt_labels_filtered = np.array(gt_labels)[valid_mask]
    filtered_indices = np.nonzero(valid_mask)[0]  # original index

    # encode string type ground truth label
    le = LabelEncoder()
    le.classes_ = np.array(uniq_types)
    gt_labels_encoded = le.fit_transform(gt_labels_filtered)

    # confusion matrix using real predicted labels as columns (fixes KeyError)
    pred_unique = np.unique(gmm_labels_filtered)
    conf_mat = confusion_matrix(
        gt_labels_encoded,
        gmm_labels_filtered,
        labels=pred_unique
    )
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    label_mapping = {pred_unique[col]: row for row, col in zip(row_ind, col_ind)}
    gmm_labels_aligned = np.array([label_mapping.get(l, -1) for l in gmm_labels_filtered])

    print("[INFO] Ground truth label classes:", le.classes_)

    # Spectral layout on filtered subgraph for plotting
    adj = load_npz("sparse_connectivity_matrix.npz")
    adj_filtered = adj[filtered_indices][:, filtered_indices]

    embedding = SpectralEmbedding(n_components=2, affinity='precomputed')
    xy = embedding.fit_transform(adj_filtered)

    coo = adj_filtered.tocoo()
    edges = list(zip(coo.row, coo.col))
    edges = [(i, j) for i, j in edges if i < j]

    # credible interval plot (kept, reading *_dict_tuned_*.npy)
    scores_ci = []
    total_runs_ci = total_runs  # typically 1 unless you increased it
    for i in range(total_runs_ci):
        fpath = f"vmog_runs/vmog_cluster_assignment_dict_tuned_{i}.npy"
        if not os.path.exists(fpath):
            print(f"[WARN] {fpath} not found, skip")
            continue
        pred_dict = np.load(fpath, allow_pickle=True).item()
        shared_ids = set(root_id_type_dict.keys()) & set(pred_dict.keys())
        if not shared_ids:
            continue
        gt_labels_arr = np.array([cluster_string_to_cid[root_id_type_dict[rid]] for rid in shared_ids])
        pred_labels_arr = np.array([pred_dict[rid] for rid in shared_ids])
        score_val, _ = compare_two_assignments(gt_labels_arr, pred_labels_arr)
        scores_ci.append(score_val)

    scores_ci = np.array(scores_ci)
    if len(scores_ci) == 0:
        print("[ERROR] no valid runs found – skip CI plot")
    else:
        mean = scores_ci.mean()
        ci_low, ci_high = np.percentile(scores_ci, [2.5, 97.5])
        print(f"\nRuns found : {len(scores_ci)}")
        print(f"Mean score : {mean:.4f}")
        print(f"95% CI     : [{ci_low:.4f}, {ci_high:.4f}]")
        eps = 1e-3
        shift_amount = -scores_ci.min() + eps if scores_ci.min() <= 0 else 0.0
        runs = np.arange(1, len(scores_ci) + 1)
        scores_shifted = scores_ci + shift_amount
        mean_shifted = scores_shifted.mean()
        ci_low_shifted, ci_high_shifted = np.percentile(scores_shifted, [2.5, 97.5])

        plt.figure(figsize=(8, 4))
        plt.plot(runs, scores_shifted, "o-", label="Individual run scores")
        plt.hlines(mean_shifted, runs[0], runs[-1], colors="C1", label=f"Mean = {mean_shifted:.4f}")
        plt.hlines([ci_low_shifted, ci_high_shifted], runs[0], runs[-1], colors="C2", linestyles="--", label="95% CI bounds")
        plt.yscale("log")
        plt.xlabel("Run index")
        plt.ylabel("Shifted score (log-scale)")
        plt.title("Run-by-run scores with 95% credible interval (log-y)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("vmog_runs/credible_interval_log.png", dpi=300)
        plt.close()

    # plots
    draw_graph(xy, edges, gt_labels_encoded, "Ground Truth Neuron Types", filename="vmog_runs/ground_truth.png")
    draw_graph(xy, edges, gmm_labels_filtered, "SBM Clustering (Before Alignment)", filename="vmog_runs/gmm_before_alignment.png")
    draw_graph(xy, edges, gmm_labels_aligned, "SBM Clustering (After Confusion Matrix Alignment)", filename="vmog_runs/gmm_after_alignment.png")

    print("[INFO] Drawing confusion matrix heatmap...")
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=pred_unique,
                yticklabels=le.classes_)
    plt.xlabel("Predicted Cluster Label (raw)")
    plt.ylabel("True Neuron Type")
    plt.title("Confusion Matrix: SBM Clustering vs Ground Truth")
    plt.tight_layout()
    plt.savefig("vmog_runs/confusion_matrix.png")
    plt.show()
    print("[INFO] Saved to vmog_runs/confusion_matrix.png")
>>>>>>> f3731bef05177212f5f802a3812cb3498d6c738e
