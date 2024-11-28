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

    # Convert W_csr to coordinate format for efficient access
    W_coo = W_csr.tocoo()

    # Initialize Adam optimizer
    optimizer = torch.optim.Adam([U], lr=lr)

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
        diff = W_reconstructed - W_batch
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

    return U


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
    U = stochastic_pca(adj_matrix, n_components, batch_size=128, lr=0.01, max_iter=1000, tol=1e-6, device=device)

    # Orthogonalize U to obtain principal components
    U_orth = orthogonalize(U)

    # Save the results
    torch.save(U.cpu(), 'U.pt')
    torch.save(U_orth.cpu(), 'U_orth.pt')
    print("Saved U and U_orth to disk.")
