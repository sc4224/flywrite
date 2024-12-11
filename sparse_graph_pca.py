import torch
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from tqdm import tqdm


def stochastic_pca(W_csr, n_components, batch_size=128, lr=0.01, max_iter=1000, tol=1e-6, device="cuda"):
    """
    Perform stochastic PCA on a sparse matrix to minimize || U U^T W - W ||^2_F.

    Args:
        W_csr: Sparse matrix (csr_matrix).
        n_components: Number of principal components (d).
        batch_size: Number of rows to sample in each iteration.
        lr: Learning rate for Adam optimizer.
        max_iter: Maximum number of iterations.
        tol: Tolerance for convergence.
        device: Device to use ("cuda" or "cpu").

    Returns:
        U: Learned principal components matrix (N x d).
    """
    print("Initializing stochastic PCA...")

    N, M = W_csr.shape  # Number of rows (N) and columns (features, M)
    d = n_components

    # Initialize U randomly
    U = torch.randn(N, d, device=device, requires_grad=True)  # Enable gradients for U
    # Initialize b randomly
    b = torch.tensor([W_csr.mean()]).to(device)

    # Convert W_csr to coordinate format for efficient access
    W_coo = W_csr.tocoo()

    # Initialize Adam optimizer
    optimizer = torch.optim.Adam([U, b], lr=lr)

    try:
        # Optimization loop
        for it in tqdm(range(max_iter)):
            # Randomly sample a batch of rows
            batch_indices = np.random.choice(N, size=batch_size, replace=False)

            # Extract the batch rows from W using SciPy's indexing
            W_batch_csr = W_csr[batch_indices]  # Still sparse
            W_batch = torch.tensor(W_batch_csr.toarray(), dtype=torch.float32, device=device)  # Dense batch
            U_batch = U[batch_indices]  # Corresponding rows of U

            # Compute reconstruction: U_batch @ (U.T @ W_batch)
            UT_W = torch.mm(W_batch, U)  # Shape: (batch_size, d)
            W_reconstructed = torch.mm(UT_W, U.T)  # Shape: (batch_size, features)

            # Compute reconstruction error for the batch
            diff = W_reconstructed - W_batch + b
            loss = torch.norm(diff, p='fro') ** 2

            # Zero the gradients
            optimizer.zero_grad()

            # Backpropagate the loss
            loss.backward()

            # Perform an Adam optimization step
            optimizer.step()

            # # Re-normalize U after the update
            # with torch.no_grad():
            #     U_batch_norm = torch.norm(U_batch, dim=1, keepdim=True)
            #     U[batch_indices] = U_batch / U_batch_norm.clamp(min=1e-8)  # Prevent division by zero

            # Check convergence (optional: compute full loss occasionally)
            if it % 100 == 0:
                print(f"Iteration {it}: Batch loss = {loss.item()}")
                if loss.item() < tol:
                    print(f"Converged at iteration {it} with batch loss = {loss.item()}")
                    break
    except KeyboardInterrupt:
        print("Interrupted by user.")

    return U, b


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

def kmeans_clustering(data, n_clusters, max_iter=100, tol=1e-4, device="cuda"):
    # Perform k-means clustering on the given data using PyTorch.
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
        if shift < tol:
            print(f"K-means converged in {i + 1} iterations with shift={shift:.6f}")
            break
        cluster_centers = new_cluster_centers

    # Calculate distances to the assigned cluster centers for each example
    distances = torch.cdist(data, cluster_centers, p=2)
    min_distances = distances.gather(1, labels.unsqueeze(1)).squeeze()

    return cluster_centers, labels, min_distances


if __name__ == "__main__":
    # Set device
    device="cpu"
    # if torch.backends.mps.is_available():
    #     device="mps"
    if torch.cuda.is_available():
        device="cuda"

    # Load the sparse matrix
    file_path = "./sparse_connectivity_matrix.npz"
    adj_matrix = load_npz(file_path)
    print(f"Loaded sparse matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

    # Perform stochastic PCA
    n_components = 32
    U, b = stochastic_pca(adj_matrix, 
                          n_components, 
                          batch_size=512, 
                          lr=0.01, 
                          max_iter=10_000,
                          tol=1e-6, 
                          device=device)
    
    # Orthogonalize U to obtain principal components
    U_orth = orthogonalize(U)

    # Perform k-means clustering
    n_clusters = 1024  # Number of clusters
    cluster_centers, labels, min_distances = kmeans_clustering(U_orth, n_clusters, device=device)

    print("Cluster centers shape:", cluster_centers.shape)
    print("Cluster labels shape:", labels.shape)
    print("Distances to cluster centers shape:", min_distances.shape)

    # build a index-to-root_id dictionary
    from index_mapping import load_mapping

    mapping = load_mapping('./root_id_to_index_mapping.json')
    rootid_mapping = dict((v, k) for k, v in mapping.items())

    # build a cluster assignment dictionary
    cluster_assignment_dict = dict()
    for i in range(len(labels)):
        root_id = mapping[i]
        cluster_assignment_dict[root_id] = labels[i].item()

    # Save the results
    torch.save(U_orth.cpu(), 'U_orth.pt')
    print("Saved U and U_orth to disk.")

    torch.save(cluster_centers.cpu(), 'pca_cluster_centers.pt')
    torch.save(labels.cpu(), 'pca_labels.pt')
    torch.save(min_distances.cpu(), 'pca_min_distances.pt')
    np.save("pca_cluster_assignment_dict.npy", cluster_assignment_dict, allow_pickle=True)