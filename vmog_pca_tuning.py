import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score

from skopt import Optimizer
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from index_mapping import load_mapping

import os
import gc

def estimate_max_jobs_per_gpu(required_per_job_mb=4000):

    if not torch.cuda.is_available():
        return 1  # CPU fallback

    total_free = 0
    for i in range(torch.cuda.device_count()):
        stats = torch.cuda.mem_get_info(i)
        free_mb = stats[0] / (1024 ** 2)
        total_free += free_mb

    max_jobs = max(int(total_free // required_per_job_mb), 1)
    return max_jobs

space = [
    Integer(10, 70, name='n_pca_components'),
    Integer(200, 800, name='k'),
    Integer(128, 1024, name='batch_size'),
    Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
    Categorical(['Adam', 'AdamW', 'SGD'], name='optimizer'),  # Optimizer choice
    Real(1e-5, 1e-2, name='reg_cov')
]

# def safe_empty_cache(device: str):
#     """
#     清空目前 process 在指定裝置上的暫存記憶體。
#     對 CUDA 額外呼叫 `torch.cuda.ipc_collect()`，把跨進程共享區塊也一併釋放。
#     """
#     if device.startswith("cuda") and torch.cuda.is_available():
#         torch.cuda.empty_cache()      # 清快取
#         torch.cuda.ipc_collect()      # 回收 IPC 共享
#     elif device.startswith("mps") and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
#         try:
#             torch.mps.empty_cache()
#         except Exception:
#             pass

def safe_empty_cache(device: str):
    """
    Clear cached memory on CUDA devices.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# def stochastic_pca(W_csr, n_pca_components=64, batch_size=512, lr=0.01, max_iter=1000, tol=1e-6, optimizer_choice="Adam",device="cuda"):
#     """
#     Perform stochastic PCA on a sparse matrix to minimize || U U^T W - W ||^2_F.
#
#     Args:
#         W_csr: Sparse matrix (csr_matrix).
#         n_pca_components: Number of principal components (d).
#         batch_size: Number of rows to sample in each iteration.
#         lr: Learning rate for Adam optimizer.
#         max_iter: Maximum number of iterations.
#         tol: Tolerance for convergence.
#         device: Device to use ("cuda" or "cpu").
#
#     Returns:
#         U: Learned principal components matrix (N x d).
#     """
#     print("Initializing stochastic PCA...")
#     print("optimizer", optimizer_choice)
#
#     N, M = W_csr.shape  # Number of rows (N) and columns (features, M)
#     d = n_pca_components
#
#     # Initialize U randomly
#     U = torch.randn(N, d, device=device, requires_grad=True)  # Enable gradients for U
#     # Initialize b randomly
#     b = torch.tensor([W_csr.mean()], dtype=torch.float32).to(device)
#
#     # Convert W_csr to coordinate format for efficient access
#     # W_coo = W_csr.tocoo()
#
#     # Initialize Adam optimizer
#     optimizer = torch.optim.Adam([U, b], lr=lr)
#     if optimizer_choice == 'Adam':
#         optimizer = torch.optim.Adam([U, b], lr=lr)
#     elif optimizer_choice == 'SGD':
#         optimizer = torch.optim.SGD([U, b], lr=lr)
#     elif optimizer_choice == 'AdamW':
#         optimizer = torch.optim.AdamW([U, b], lr=lr)
#
#     try:
#         # Optimization loop
#         for it in tqdm(range(max_iter)):
#             # Randomly sample a batch of rows
#             batch_indices = np.random.choice(N, size=batch_size, replace=False)
#
#             # Extract the batch rows from W using SciPy's indexing
#             W_batch_csr = W_csr[batch_indices]  # Still sparse
#             W_batch = torch.tensor(W_batch_csr.toarray(), dtype=torch.float32, device=device)  # Dense batch
#             # U_batch = U[batch_indices]  # Corresponding rows of U
#
#             # Compute reconstruction: U_batch @ (U.T @ W_batch)
#             UT_W = torch.mm(W_batch, U)  # Shape: (batch_size, d)
#             W_reconstructed = torch.mm(UT_W, U.T)  # Shape: (batch_size, features)
#
#             # Compute reconstruction error for the batch
#             diff = W_reconstructed - W_batch + b
#             loss = torch.norm(diff, p='fro') ** 2
#
#             # Zero the gradients
#             optimizer.zero_grad()
#
#             # Backpropagate the loss
#             loss.backward()
#
#             torch.nn.utils.clip_grad_norm_([U, b], max_norm=1.0)
#
#
#             # Perform an Adam optimization step
#             optimizer.step()
#             del batch_indices, W_batch_csr, W_batch, UT_W, W_reconstructed, diff
#
# #             # Re-normalize U after the update
# #            with torch.no_grad():
# #                U_batch_norm = torch.norm(U_batch, dim=1, keepdim=True)
# #                U[batch_indices] = U_batch / U_batch_norm.clamp(min=1e-8)  # Prevent division by zero
#
#             # Check convergence (optional: compute full loss occasionally)
#             if it % 100 == 0:
#                 gc.collect()
#                 safe_empty_cache(device)
#                 print(f"Iteration {it}: Batch loss = {loss.item()}")
#                 if loss.item() < tol:
#                     print(f"Converged at iteration {it} with batch loss = {loss.item()}")
#                     break
#     except KeyboardInterrupt:
#         print("Interrupted by user.")
#
#     del W_csr, N, M, d
#     gc.collect()
#     safe_empty_cache(device)
#
#
#     return U, b

def stochastic_pca(W_csr, n_pca_components=64, batch_size=512, lr=0.01, max_iter=1000, tol=1e-6, optimizer_choice="Adam", device="cuda"):
    """
    Safe + memory-efficient stochastic PCA with sparse input matrix W (CSR),
    dense learnable U (N × d), and sparse matmul.

    Args:
        W_csr: Scipy CSR sparse matrix (N × N)
        n_pca_components: Output dimension d
        batch_size: SGD batch size
        lr: Learning rate
        max_iter: Max iterations
        tol: Tolerance for early stopping
        device: "cuda" or "cpu"

    Returns:
        U: Torch tensor (N × d)
    """

    N, M = W_csr.shape
    assert N == M, "W must be square (symmetric adj matrix)"
    d = n_pca_components

    # Learnable dense embedding U
    U = torch.randn(N, d, device=device, requires_grad=True)

    # Initialize bias (optional)
    b = torch.tensor([W_csr.mean()], dtype=torch.float32, device=device)

    # Optimizer
    if optimizer_choice == 'Adam':
        optimizer = torch.optim.Adam([U, b], lr=lr)
    elif optimizer_choice == 'SGD':
        optimizer = torch.optim.SGD([U, b], lr=lr)
    elif optimizer_choice == 'AdamW':
        optimizer = torch.optim.AdamW([U, b], lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    try:
        for it in tqdm(range(max_iter)):
            # Randomly select batch row indices
            batch_indices = np.random.choice(N, size=batch_size, replace=False)
            W_batch_csr = W_csr[batch_indices]  # shape: (B × N)

            # Convert to PyTorch sparse COO tensor
            W_batch_coo = W_batch_csr.tocoo()
            indices = torch.tensor(np.vstack((W_batch_coo.row, W_batch_coo.col)), dtype=torch.long, device=device)
            values = torch.tensor(W_batch_coo.data, dtype=torch.float32, device=device)
            W_sparse = torch.sparse_coo_tensor(indices, values, size=W_batch_csr.shape).coalesce()

            # Gather corresponding U
            U_batch = U[batch_indices]         # shape: (B × d)
            U_all   = U                        # shape: (N × d)

            # Compute: W_hat = U_batch @ U_all.T
            # WT_U = torch.matmul(U_all.T, U_batch.T).T   # shape: (B × d)
            # WT_U = torch.matmul(U_batch, U_all.T)
            W_recon = torch.matmul(U_batch, U_all.T)
            # W_recon = torch.matmul(WT_U, U_all.T)       # shape: (B × N)

            # Only compute reconstruction loss on non-zero entries of W
            pred_values = W_recon[W_batch_coo.row, W_batch_coo.col]
            loss = torch.mean((pred_values - values + b)**2)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([U], max_norm=1.0)
            optimizer.step()

            if it % 100 == 0:
                print(f"[Iter {it}] Loss: {loss.item():.6f}")
                if loss.item() < tol:
                    print(f"Converged early at iter {it}")
                    break

            # Clean up
            del W_batch_csr, W_batch_coo, indices, values, W_sparse, W_recon, pred_values
            gc.collect()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    except KeyboardInterrupt:
        print("Training interrupted.")

    return U.detach(), b.detach()


def orthogonalize(U):
    """
    Orthogonalize the matrix U to obtain principal components.
    Args:
        U: Matrix of shape (N, d) where d is the number of components.

    Returns:
        U_orth: Orthogonalized U of shape (N, d).
    """
    print("Orthogonalizing U...")
    Q, _ = torch.linalg.qr(U)  # QR decomposition for orthogonalization
    return Q

class BatchedEMGaussianMixture(nn.Module):
    def __init__(self, n_components, n_features, max_iter=1000, tol=1e-3, reg_covar=1e-6, init_strategy='kmeans',device='cpu'):
        """
        PyTorch implementation of Gaussian Mixture Model with batched EM algorithm.

        Args:
            n_components: Number of mixture components
            n_features: Number of features/dimensions in the data
            init_strategy: Initialization strategy ('kmeans')
            max_iter: Maximum number of EM iterations
            tol: Tolerance for convergence
            reg_covar: Regularization added to covariance matrices
            device: Device to use ('cpu', 'cuda', or 'mps')
        """
        super(BatchedEMGaussianMixture, self).__init__()

        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_strategy = init_strategy
        self.device = device

        # Learnable parameters (initialized in initialize_parameters)
        # These are buffers, not parameters, as we update them via EM not gradient descent
        self.register_buffer('means_', torch.zeros(n_components, n_features, device=device))
        self.register_buffer('covs_', torch.zeros(n_components, n_features, n_features, device=device))
        self.register_buffer('weights_', torch.ones(n_components, device=device) / n_components)

        # For diagonal covariance (more efficient)
        self.register_buffer('log_vars_', torch.zeros(n_components, n_features, device=device))

        # For tracking
        self.lower_bound_history_ = []
        self.n_iter_ = 0
        self.converged_ = False

        # For results
        self.responsibilities_ = None
        self.labels_ = None

    def initialize_parameters(self, X):
        """Initialize GMM parameters, optionally using k-means."""
        n_samples = X.shape[0]

        # initialization
        if self.init_strategy == 'kmeans':
            from sklearn.cluster import KMeans
            X_np = X.cpu().numpy()

            if np.isnan(X_np).any():
                raise ValueError("X contains NaN values, cannot run KMeans.")

            if X_np.shape[0] < self.n_components:
                raise ValueError("Number of samples is less than number of clusters.")

            kmeans = KMeans(n_clusters=self.n_components, n_init=5, random_state=0).fit(X_np)

            if not hasattr(kmeans, 'cluster_centers_'):
                raise RuntimeError("KMeans failed to compute cluster_centers_.")

            self.means_ = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=self.device)
            X_subset = X  # 用來後面估計 variance
        else:
            X_subset = X
            # Initialize means with random data points
            indices = torch.randperm(len(X_subset))[:self.n_components]
            self.means_ = X_subset[indices].clone()

        # Initialize with uniform weights
        self.weights_ = torch.ones(self.n_components, device=self.device) / self.n_components

        # Estimate initial variances from data
        data_var = torch.var(X_subset, dim=0)
        self.log_vars_ = torch.log(data_var + self.reg_covar).repeat(self.n_components, 1)

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if self.device == 'cuda' else None
        safe_empty_cache(self.device)
        # torch.mps.empty_cache() if (hasattr(torch, 'mps') and self.device == 'mps') else None

    def e_step(self, X):
        """E-step: Compute responsibilities (posterior probabilities)."""
        n_samples = X.shape[0]
        log_resp = torch.zeros(n_samples, self.n_components, device=X.device)

        # Compute log probabilities for each component
        for k in range(self.n_components):
            # Using diagonal covariance for efficiency
            vars_k = torch.exp(self.log_vars_[k])

            # Compute log probabilities efficiently
            diff = X - self.means_[k]
            log_prob = -0.5 * (
                torch.sum(torch.log(2 * np.pi * vars_k)) +
                torch.sum(diff**2 / vars_k.unsqueeze(0), dim=1)
            )

            log_resp[:, k] = torch.log(self.weights_[k] + 1e-10) + log_prob
            del diff, log_prob

        # Normalize (log-sum-exp trick for numerical stability)
        log_resp_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)
        log_resp = log_resp - log_resp_norm

        # Convert to probabilities
        resp = torch.exp(log_resp)

        # Clean up device tensors
        del log_resp, log_resp_norm

        # Force memory cleanup
        torch.cuda.empty_cache() if self.device == 'cuda' else None
        safe_empty_cache(self.device)
        # torch.mps.empty_cache() if (hasattr(torch, 'mps') and self.device == 'mps') else None

        return resp

    def compute_lower_bound(self, X, resp):
        """Compute the lower bound (ELBO) for current parameters."""
        n_samples = X.shape[0]
        lower_bound = 0.0

        # Log-likelihood contribution
        for k in range(self.n_components):
            vars_k = torch.exp(self.log_vars_[k])

            diff = X - self.means_[k]
            log_prob = -0.5 * (
                torch.sum(torch.log(2 * np.pi * vars_k)) +
                torch.sum(diff**2 / vars_k.unsqueeze(0), dim=1)
            )

            lower_bound += torch.sum(
                resp[:, k] * (torch.log(self.weights_[k] + 1e-10) + log_prob)
            )
            del vars_k, diff, log_prob

        # Entropy contribution
        entropy = -torch.sum(resp * torch.log(resp + 1e-10))
        lower_bound += entropy

        # Clean up device tensors
        del entropy
        # Force cleanup
        torch.cuda.empty_cache() if self.device == 'cuda' else None
        safe_empty_cache(self.device)
        # torch.mps.empty_cache() if (hasattr(torch, 'mps') and self.device == 'mps') else None

        return lower_bound / n_samples

    def fit(self, X, batch_size=1024, verbose=False):
        """
        Fit the GMM using batched EM algorithm.

        Args:
            X: Input data tensor of shape (n_samples, n_features)
            batch_size: Size of batches for processing
            kmeans: Optional k-means result for initialization
            verbose: Whether to print progress

        Returns:
            self: Fitted model
        """
        n_samples = X.shape[0]

        # Move data to the right device if needed
        if X.device != self.device:
            X = X.to(self.device)

        # Initialize parameters
        self.initialize_parameters(X)

        # Store for convergence check
        prev_lower_bound = -np.inf
        prev_means = self.means_.clone()

        # Create data loader for batched processing
        dataset = TensorDataset(X)
        batch_size = int(batch_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for iteration in range(self.max_iter):
            # Initialize accumulators for sufficient statistics
            nk = torch.zeros(self.n_components, device=self.device)
            means_numerator = torch.zeros_like(self.means_)
            vars_numerator = torch.zeros_like(self.log_vars_)

            # For ELBO calculation
            total_lower_bound = 0.0

            # Process batches
            n_processed = 0
            for batch_idx, (batch_X,) in enumerate(tqdm(loader, desc=f"EM Iteration {iteration+1}")):
                batch_size_actual = batch_X.shape[0]
                n_processed += batch_size_actual

                # E-step: compute responsibilities for this batch
                with torch.no_grad():
                    batch_resp = self.e_step(batch_X)

                # Accumulate statistics for M-step
                batch_nk = torch.sum(batch_resp, dim=0)
                nk += batch_nk

                # Means numerator: Σ_i r_ik * x_i
                means_numerator += torch.matmul(batch_resp.T, batch_X)

                # Variances numerator: Σ_i r_ik * (x_i - μ_k)^2
                for k in range(self.n_components):
                    diff = batch_X - self.means_[k]
                    # Weighted sum of squared differences
                    weighted_diff_sq = batch_resp[:, k].unsqueeze(1) * diff**2
                    vars_numerator[k] += torch.sum(weighted_diff_sq, dim=0)
                    del diff, weighted_diff_sq

                # Contribution to lower bound
                batch_lower_bound = self.compute_lower_bound(batch_X, batch_resp)
                total_lower_bound += batch_lower_bound * batch_size_actual

                # Clean up between batches
                del batch_resp, batch_nk
                # if hasattr(torch, 'mps') and torch.backends.mps.is_available() and self.device == 'mps':
                #     torch.mps.empty_cache()
                if torch.cuda.is_available() and self.device == 'cuda':
                    torch.cuda.empty_cache()
                else:
                    safe_empty_cache(self.device)

            # M-step: update parameters using accumulated statistics
            with torch.no_grad():
                # Update means: μ_k = (Σ_i r_ik * x_i) / (Σ_i r_ik)
                self.means_ = means_numerator / nk.unsqueeze(1)

                # Update variances: σ²_k = (Σ_i r_ik * (x_i - μ_k)²) / (Σ_i r_ik)
                self.log_vars_ = torch.log(vars_numerator / nk.unsqueeze(1) + self.reg_covar)

                # Update weights: π_k = (Σ_i r_ik) / N
                self.weights_ = nk / n_samples

            # Normalize lower bound
            total_lower_bound /= n_samples
            self.lower_bound_history_.append(total_lower_bound.item())

            if verbose:
                print(f"Iteration {iteration+1}: Lower bound = {total_lower_bound.item():.4f}")

            # Check for convergence
            if iteration > 0:
                mean_change = torch.mean(torch.abs(self.means_ - prev_means))
                lb_change = total_lower_bound - prev_lower_bound

                if verbose:
                    print(f"Mean change: {mean_change.item():.6f}, LB change: {lb_change.item():.6f}")

                if (mean_change < self.tol or lb_change < self.tol) and lb_change >= 0:
                    self.converged_ = True
                    self.n_iter_ = iteration + 1
                    if verbose:
                        print(f"Converged after {self.n_iter_} iterations")
                    break

            prev_lower_bound = total_lower_bound
            prev_means = self.means_.clone()

        # If not converged, set final iteration count
        if not self.converged_:
            self.n_iter_ = self.max_iter
            if verbose:
                print(f"Did not converge after {self.max_iter} iterations")

        # Compute final responsibilities and labels in batches
        self.responsibilities_ = torch.zeros(n_samples, self.n_components, device='cpu')
        self.labels_ = torch.zeros(n_samples, dtype=torch.long, device='cpu')

        with torch.no_grad():
            start_idx = 0
            for batch_idx, (batch_X,) in enumerate(tqdm(loader, desc="Computing final assignments")):
                batch_size_actual = batch_X.shape[0]
                batch_resp = self.e_step(batch_X)

                # Move to CPU for storage (to save GPU memory)
                self.responsibilities_[start_idx:start_idx+batch_size_actual] = batch_resp.cpu()
                self.labels_[start_idx:start_idx+batch_size_actual] = torch.argmax(batch_resp, dim=1).cpu()

                start_idx += batch_size_actual

        # Analysis of clustering results
        unique_labels, counts = torch.unique(self.labels_, return_counts=True)
        if verbose:
            print(f"Found {len(unique_labels)} unique clusters out of {self.n_components} components")
            #for i, label in enumerate(unique_labels):
            #    print(f"Cluster {label.item()}: {counts[i].item()} samples ({counts[i].item()/n_samples*100:.2f}%)")

        return self

@use_named_args(space)
def objective(**params):
    return run_objective_with_device(params, device="cuda")
    #run = wdb.init(
    #    project="flywrite",
    #)
def run_objective_with_device(params, device="cuda"):
    # 原本 objective 的邏輯搬到這裡
    torch.cuda.set_device(device)

    n_pca_components =int(params["n_pca_components"])
    n_mog_components = int(params["k"])
    learning_rate = float(params["learning_rate"])
    pca_epochs = 1000
    batch_size = int(params["batch_size"])
    optimizer_choice = params["optimizer"]
    reg_cov = float(params["reg_cov"])

    # Load the sparse matrix
    file_path = "./sparse_connectivity_matrix.npz"
    adj_matrix = load_npz(file_path)
    print(f"Loaded sparse matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

    U, b = stochastic_pca(adj_matrix,
                      n_pca_components,
                      batch_size=batch_size,
                      lr=learning_rate,
                      max_iter=pca_epochs,
                      tol=1e-6,
                      optimizer_choice=optimizer_choice,
                      device=device)

    # Orthogonalize U to obtain principal components
    U_orth = orthogonalize(U).detach()

    del U, b# adj_matrix
    gc.collect()
    # torch.mps.empty_cache()
    safe_empty_cache(device)

    # cluster_centers, labels, min_distances = kmeans_clustering(U_orth, n_clusters, device=device)
    # mog_model = VariationalMixtureOfGaussians(input_dim=n_pca_components, n_components=n_mog_components).to(device)
    gmm_model = BatchedEMGaussianMixture(
        n_components=n_mog_components,
        n_features=n_pca_components,
        max_iter=1000,  # EM might need more iterations
        tol=1e-4,
        reg_covar=reg_cov,
        device=device,
        init_strategy = 'kmeans',
    )

    # Fit the model
    gmm_model.fit(U_orth, batch_size=batch_size, verbose=True)

    # Get labels - already computed during fitting
    labels = gmm_model.labels_

    # === ➊ 計算 BIC =======================================
    n_samples   = U_orth.shape[0]
    n_clusters  = n_mog_components
    n_features  = n_pca_components

    # 參數個數：權重(k-1) + 均值(k·d) + 對角變異數(k·d)
    n_params = (n_clusters - 1) + 2 * n_clusters * n_features

    # gmm_model.lower_bound_history_ 裡存的是每回合的 ELBO / sample
    log_likelihood = (
        gmm_model.lower_bound_history_[-1] * n_samples
        if len(gmm_model.lower_bound_history_) > 0 else -np.inf
    )

    bic_score = -2 * log_likelihood + n_params * np.log(n_samples)
    print(f"BIC score: {bic_score:.2f}")

    #U_orth_np = U_orth.cpu().numpy()
    #labels_np = labels.numpy()

    # Compute the silhouette score.
    #score = silhouette_score(U_orth_np, labels_np)

    # Sample a subset of points
    sample_size = 5000

    if adj_matrix.shape[0] <= sample_size:
        sample_indices = np.arange(adj_matrix.shape[0])
    else:
        sample_indices = np.random.choice(adj_matrix.shape[0], size=sample_size, replace=False)

    adj_sample = adj_matrix[sample_indices]  # 仍是 CSR，shape = (k, N_full)
    labels_sample = labels[sample_indices].cpu().numpy()

    non_zero_mask = adj_sample.getnnz(axis=1) > 0            # 剔除全 0 列
    adj_sample    = adj_sample[non_zero_mask]
    labels_sample = labels_sample[non_zero_mask]

    # Check if valid number of clusters in sample
    if len(np.unique(labels_sample)) < 2:
        print("Not enough clusters in sample, returning worst score")
        return 1.0  # silhouette score is minimized, so return 1.0

    score = silhouette_score(
        adj_sample,  # CSR matrix
        labels_sample,
        metric="euclidean",  # cosine/euclidean 都支援 CSR；cosine 對高度稀疏較穩
        n_jobs=1  # 避免在小樣本時多執行緒 overhead
    )

    del adj_sample, labels_sample, sample_indices
    gc.collect()
    safe_empty_cache(device)

    return -score


    # del labels, U_orth
    # gc.collect()
    # # torch.mps.empty_cache()
    # safe_empty_cache(device)
    # # Since gp_minimize minimizes the objective, return the negative silhouette score.
    # return -score

def run_pca_gmm(batch, index, best_params, device="cuda"):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device)
    iteration = (batch * 4) + index
    n_components = int(best_params["n_pca_components"])
    n_clusters = int(best_params["k"])

    # Load the sparse matrix
    file_path = "./sparse_connectivity_matrix.npz"
    adj_matrix = load_npz(file_path)
    print(f"Loaded sparse matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")


    U, b = stochastic_pca(adj_matrix,
                      n_components,
                      batch_size=int(best_params["batch_size"]),
                      lr=float(best_params["learning_rate"]),
                      max_iter=1000,
                      tol=1e-6,
                      optimizer_choice=best_params["optimizer"],
                      device=device)

    # Orthogonalize U to obtain principal components
    U_orth = orthogonalize(U).detach()

    del U, b, adj_matrix
    gc.collect()
    # torch.mps.empty_cache()
    safe_empty_cache(device)


    # cluster_centers, labels, min_distances = kmeans_clustering(U_orth, n_clusters, device=device)
    # mog_model = VariationalMixtureOfGaussians(input_dim=n_pca_components, n_components=n_mog_components).to(device)
    gmm_model = BatchedEMGaussianMixture(
        n_components=n_clusters,
        n_features=int(best_params["n_pca_components"]),
        max_iter=1000,  # EM might need more iterations
        tol=1e-4,
        reg_covar=float(best_params["reg_cov"]),
        device=device
    )

    # Fit the model
    gmm_model.fit(U_orth, batch_size=int(best_params["batch_size"]), verbose=True)

    # Get labels - already computed during fitting
    labels = gmm_model.labels_
    cluster_centers = gmm_model.means_

    if labels != None and cluster_centers != None:
        print("Cluster centers shape:", cluster_centers.shape)
        print("Cluster labels shape:", labels.shape)

        mapping = load_mapping('./root_id_to_index_mapping.json')

       # build a cluster assignment dictionary
        cluster_assignment_dict = dict()
        for i in range(len(labels)):
            root_id = mapping[i]
            cluster_assignment_dict[root_id] = labels[i].item()

        # 在儲存檔案之前，先確保資料夾存在
        os.makedirs("vmog_runs", exist_ok=True)
        # Save the results
        torch.save(U_orth.cpu(), f"vmog_runs/U_orth_tuned_{iteration}.pt")
        print("Saved U and U_orth to disk.")

        torch.save(cluster_centers.cpu(), f"vmog_runs/vmog_cluster_centers_tuned_{iteration}.pt")
        torch.save(labels.cpu(), f"vmog_runs/vmog_labels_tuned_{iteration}.pt")
        np.save(f"vmog_runs/vmog_cluster_assignment_dict_tuned_{iteration}.npy", cluster_assignment_dict, allow_pickle=True)

        del labels, U_orth, mapping, cluster_assignment_dict, cluster_centers
    gc.collect()
    # torch.mps.empty_cache()
    safe_empty_cache(device)

if __name__ == "__main__":
    
    # Set device
    device="cpu"

    if torch.cuda.is_available():
        device="cuda:0"

    # === Smoke Test (Ultra Debug) ===
    print("[DEBUG] Running ultra-fast smoke test...")

    debug_params = {
        "n_pca_components": 64,
        "k": 3,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "reg_cov": 1e-5
    }

    try:
        _ = run_objective_with_device(debug_params, device)
        print("[DEBUG] ✅ Smoke test passed: No crash.")
    except Exception as e:
        print("[DEBUG] ❌ Smoke test failed:")
        print(e)
        import traceback
        traceback.print_exc()
        exit(1)

    # Step 2: 總共要跑的 trial 數
    total_trials = 48
    required_per_job_mb = 4000  # 每個 job 預估用多少 GPU 記憶體（你可以調）

    n_jobs_per_batch = estimate_max_jobs_per_gpu(required_per_job_mb)
    print(f"[INFO] Max parallel jobs based on GPU memory: {n_jobs_per_batch}")

    n_batches = (total_trials + n_jobs_per_batch - 1) // n_jobs_per_batch
    print(f"[INFO] Total batches needed: {n_batches} to complete {total_trials} trials.")

    opt = Optimizer(dimensions=space, base_estimator="GP", acq_func="EI", random_state=42 )

    # --- 新增：把維度名稱先存起來，只建立一次即可 ---
    space_names = [d.name for d in space]  # ['n_pca_components', 'k', …]

    for batch_idx in range(n_batches):
        candidates = opt.ask(n_points=n_jobs_per_batch)
        if candidates is None:
            raise ValueError("opt.ask() returned None")

        scores = Parallel(n_jobs=n_jobs_per_batch)(
            delayed(run_objective_with_device)(
                {name: val for name, val in zip(space_names, point)},
                device="cuda:0"  # 或你可以交錯設 cuda:0, cuda:1...
            )
            for point in candidates
        )

        opt.tell(candidates, scores)
        print(f"Batch {batch_idx + 1}/{n_batches}: Best score = {-min(opt.yi):.4f}")

    # for i in range(n_batches):
    #    candidates = opt.ask(n_points=n_jobs_per_batch)
    #
    #    if candidates is None:
    #        raise ValueError("opt.ask() returned None")
    #
    #    n_gpu = torch.cuda.device_count()
    #    scores = Parallel(n_jobs=n_jobs_per_batch)(
    #        delayed(objective)(
    #            params,
    #            device=f"cuda:{i % n_gpu}"  # 0,1,0,1 輪流
    #        )
    #        for i, params in enumerate(candidates)
    #    )
    #
    #    opt.tell(candidates, scores)
    #    print(f"All scores so far = {opt.yi}")
    #    print(f"Batch {i+1}: Best score so far = {-min(opt.yi):.4f}")

    # Best config
    best_idx = np.argmin(opt.yi)
    best_params = {
        dim.name: int(val) if isinstance(val, (np.integer, float)) and isinstance(dim, Integer) else val
        for dim, val in zip(space, opt.Xi[best_idx])
    }
    print("\nBest configuration:")
    print(f"  Params: {best_params}")
    print(f"  Silhouette Score: {-opt.yi[best_idx]:.4f}")

    #TODO: Implement credible intervals
    for batch in range(n_batches):
        n_gpu = torch.cuda.device_count()  # 可用 GPU 數（例如 2）

        _ = Parallel(n_jobs=n_jobs_per_batch)(
            delayed(run_pca_gmm)(
                batch,  # 第幾個 batch
                idx + 1,  # index 由 1 開始
                best_params,  # 最佳超參數 dict
                device=f"cuda:{idx % n_gpu}" if n_gpu else "cpu"  # 0,1,0,1… 輪流分配
            )
            for idx in range(n_jobs_per_batch)
        )

