import torch
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from tqdm import tqdm

def sparse_pca(W_csr, n_components, max_iter=100, tol=1e-6, device="cuda"):
    """Perform PCA on a sparse matrix using PyTorch without densifying."""
    print("Starting PCA...")

    # Compute column means without densifying the sparse matrix
    col_means = torch.tensor(W_csr.mean(axis=0).A1, dtype=torch.float32, device=device)

    # Convert sparse matrix to PyTorch sparse tensor
    coo = W_csr.tocoo()
    indices = torch.stack((torch.tensor(coo.row, dtype=torch.long),
                           torch.tensor(coo.col, dtype=torch.long)))
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    W_sparse = torch.sparse_coo_tensor(indices, values, size=W_csr.shape, device=device)

    n_features = W_csr.shape[1]
    components = torch.randn((n_components, n_features), device=device)
    components = torch.nn.functional.normalize(components, dim=1)

    for i in tqdm(range(max_iter)):
        # Subtract column means on-the-fly during sparse-dense multiplication
        scores = torch.sparse.mm(W_sparse, components.T)
        scores -= torch.outer(torch.ones(scores.size(0), device=device), col_means @ components.T)

        new_components = torch.sparse.mm(W_sparse.T, scores).T
        new_components -= torch.outer(torch.ones(new_components.size(0), device=device), col_means @ scores.T)
        new_components = torch.nn.functional.normalize(new_components, dim=1)

        # Check for convergence
        diff = torch.norm(new_components - components, p='fro').item()
        print(f"Iteration {i + 1}: diff={diff:.6f}")
        if diff < tol:
            print(f"Converged in {i + 1} iterations with diff={diff:.6f}")
            break

        components = new_components

    explained_variance = torch.sum(scores ** 2, dim=0)
    return components, explained_variance, col_means

def kmeans_clustering(data, n_clusters, max_iter=100, tol=1e-4, device="cuda"):
    """Perform k-means clustering on dense data."""
    print("Starting K-Means...")
    n_samples, n_features = data.shape
    data = data.to(device)
    indices = torch.randint(0, n_samples, (n_clusters,), device=device)
    cluster_centers = data[indices]

    for i in range(max_iter):
        distances = torch.cdist(data, cluster_centers, p=2)
        labels = torch.argmin(distances, dim=1)
        new_cluster_centers = torch.stack([
            data[labels == k].mean(dim=0) if (labels == k).sum() > 0 else cluster_centers[k]
            for k in range(n_clusters)
        ])
        shift = torch.norm(new_cluster_centers - cluster_centers, p='fro').item()
        print(f"Iteration {i + 1}: Shift={shift:.6f}")

        if shift < tol:
            print(f"K-means converged in {i + 1} iterations with shift={shift:.6f}")
            break

        cluster_centers = new_cluster_centers

    # Calculate distances to the assigned cluster centers for each example
    distances = torch.cdist(data, cluster_centers, p=2)
    min_distances = distances.gather(1, labels.unsqueeze(1)).squeeze()

    return cluster_centers, labels, min_distances

# Set device priority
device = "cpu"  # Default to CPU
if torch.cuda.is_available():
    device = "cuda"

# Load the sparse matrix
adj_matrix = load_npz("./sparse_connectivity_matrix.npz")
adj_matrix.data = (adj_matrix.data > 0).astype(np.int8)
print(f"Loaded adjacency matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

# Perform PCA
n_components = 32
components, explained_variance, col_means = sparse_pca(adj_matrix, n_components, device=device)
print("Components shape:", components.shape)
print("Explained variance:", explained_variance)

# Perform k-means clustering
n_clusters = 64  # Number of clusters
cluster_centers, labels, min_distances = kmeans_clustering(components.T, n_clusters, device=device)

print("Cluster centers shape:", cluster_centers.shape)
print("Cluster labels shape:", labels.shape)
print("Distances to cluster centers shape:", min_distances.shape)

# Save the results
torch.save(components.cpu(), 'components.pt')
torch.save(explained_variance.cpu(), 'explained_variance.pt')
torch.save(cluster_centers.cpu(), 'cluster_centers.pt')
torch.save(labels.cpu(), 'labels.pt')
torch.save(min_distances.cpu(), 'min_distances.pt')
torch.save(col_means.cpu(), 'col_means.pt')