import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.sparse import load_npz
from tqdm import tqdm

def stochastic_pca(W_csr, n_components, batch_size=128, lr=0.01, max_iter=1000, tol=1e-6, optimizer_choice="Adam",device="cuda"):
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
    print("optimizer", optimizer_choice)

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
    if optimizer_choice == 'Adam':
        optimizer = torch.optim.Adam([U, b], lr=lr)
    elif optimizer_choice == 'SGD':
        optimizer = torch.optim.SGD([U, b], lr=lr)
    elif optimizer_choice == 'AdamW':
        optimizer = torch.optim.AdamW([U, b], lr=lr)

    try:
        # Optimization loop
        for it in tqdm(range(max_iter)):
            # Randomly sample a batch of rows
            batch_indices = np.random.choice(N, size=batch_size, replace=False)

            # Extract the batch rows from W using SciPy's indexing
            W_batch_csr = W_csr[batch_indices]  # Still sparse
            W_batch = torch.tensor(W_batch_csr.toarray(), dtype=torch.float32, device=device)  # Dense batch
            # U_batch = U[batch_indices]  # Corresponding rows of U

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

            torch.nn.utils.clip_grad_norm_([U, b], max_norm=1.0)


            # Perform an Adam optimization step
            optimizer.step()

#             # Re-normalize U after the update
#            with torch.no_grad():
#                U_batch_norm = torch.norm(U_batch, dim=1, keepdim=True)
#                U[batch_indices] = U_batch / U_batch_norm.clamp(min=1e-8)  # Prevent division by zero

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

#############################################
# 2. Variational Mixture of Gaussians Model #
#############################################

class VariationalMixtureOfGaussians(nn.Module):
    def __init__(self, input_dim, n_components):
        """
        Variational Mixture of Gaussians.
        
        Args:
            input_dim (int): Dimensionality of the input (should match PCA output).
            n_components (int): Number of mixture components.
        """
        super(VariationalMixtureOfGaussians, self).__init__()
        self.n_components = n_components
        # Unnormalized log mixture weights
        self.logits = nn.Parameter(torch.zeros(n_components))
        # Means for each component (shape: n_components x input_dim)
        self.means = nn.Parameter(torch.randn(n_components, input_dim))
        # Log-variances for each component (shape: n_components x input_dim)
        self.log_vars = nn.Parameter(torch.zeros(n_components, input_dim))
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size, input_dim = x.shape
        # Expand x to shape (batch_size, 1, input_dim)
        x_expanded = x.unsqueeze(1)
        # Expand means and log_vars to shape (1, n_components, input_dim)
        means = self.means.unsqueeze(0)
        log_vars = self.log_vars.unsqueeze(0)
        
        # Compute log probability for each component per data point.
        c = torch.tensor(2 * np.pi, device=x.device, dtype=x.dtype)
        log_prob = -0.5 * (torch.log(c) + log_vars + ((x_expanded - means) ** 2) / torch.exp(log_vars))
        log_prob = log_prob.sum(dim=2)  # shape: (batch_size, n_components)
        
        # Add log mixture weights
        log_mix = F.log_softmax(self.logits, dim=0)  # shape: (n_components,)
        log_mix = log_mix.unsqueeze(0)  # shape: (1, n_components)
        log_prob = log_prob + log_mix  # shape: (batch_size, n_components)
        
        # Marginalize over mixture components using log-sum-exp
        log_likelihood = torch.logsumexp(log_prob, dim=1)  # shape: (batch_size,)
        return log_likelihood
    
    def loss(self, x):
        # Negative marginal log likelihood (to be minimized)
        log_likelihood = self.forward(x)
        return -log_likelihood.mean()

#######################
# 3. Putting It Together #
#######################

if __name__ == "__main__":
    # For reproducibility and device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # Generate synthetic data (or load your dataset)
    # For example, 1000 samples, 50 features.
    N, D = 1000, 50
    X = torch.randn(N, D, device=device)

    # --- Step 1: Run Stochastic PCA ---
    n_pca_components = 10  # Reduce from 50 to 10 dimensions

    file_path = "./sparse_connectivity_matrix.npz"

    adj_matrix = load_npz(file_path)

    U, b = stochastic_pca(adj_matrix, 
                      n_pca_components, 
                      batch_size=512, 
                      lr=0.01, 
                      max_iter=100,
                      tol=1e-6,
                      optimizer_choice="Adam",
                      device=device)

    U_orth = orthogonalize(U).detach()

    # --- Step 2: Train Variational Mixture of Gaussians on PCA output ---
    n_mog_components = 3  # Number of clusters/components in the mixture
    mog_model = VariationalMixtureOfGaussians(input_dim=n_pca_components, n_components=n_mog_components).to(device)
    optimizer_mog = optim.Adam(mog_model.parameters(), lr=0.01)

    n_epochs_mog = 200
    for epoch in range(n_epochs_mog):
        optimizer_mog.zero_grad()
        loss = mog_model.loss(U_orth)
        loss.backward()
        optimizer_mog.step()
        if epoch % 20 == 0:
            print(f"MOG Epoch {epoch}, Loss: {loss.item():.4f}")

    # After training, inspect the learned parameters
    with torch.no_grad():
        learned_weights = F.softmax(mog_model.logits, dim=0)
        learned_means = mog_model.means
        learned_vars = torch.exp(mog_model.log_vars)

        print("Learned mixture weights:", learned_weights.cpu().numpy())
        print("Learned means:", learned_means.detach().cpu().numpy())
        print("Learned variances:", learned_vars.detach().cpu().numpy())

        # Compute responsibilities for each sample in U_orth.
        # U_orth is assumed to have shape (N, d), where d == n_pca_components.
        N, d = U_orth.shape
        # Expand U_orth to (N, 1, d)
        U_expanded = U_orth.unsqueeze(1)  # shape: (N, 1, d)
        # Expand means and log_vars from mog_model:
        means = mog_model.means.unsqueeze(0)      # shape: (1, n_components, d)
        log_vars = mog_model.log_vars.unsqueeze(0)  # shape: (1, n_components, d)
        
        # Compute the log probability of each sample under each component.
        # Using: log p(x|z=k) = -0.5 * [ log(2*pi) + log(var) + ((x - μ)^2 / var) ]
        c = torch.tensor(2 * np.pi, device=U_orth.device, dtype=U_orth.dtype)
        log_prob = -0.5 * (torch.log(c) + log_vars + ((U_expanded - means) ** 2) / torch.exp(log_vars))
        # Sum over the feature dimension to get (N, n_components)
        log_prob = log_prob.sum(dim=2)
        
        # Add log mixture weights (normalize with log_softmax)
        log_mix = F.log_softmax(mog_model.logits, dim=0).unsqueeze(0)  # shape: (1, n_components)
        log_joint = log_prob + log_mix  # shape: (N, n_components)
        
        # Normalize using log-sum-exp to get log responsibilities
        log_resp = log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)  # (N, n_components)
        # Exponentiate to get responsibilities (posterior probabilities)
        resp = torch.exp(log_resp)  # shape: (N, n_components)
        
        # Hard assignments: assign each sample to the component with highest responsibility
        labels = torch.argmax(resp, dim=1)  # shape: (N,)

        # build a index-to-root_id dictionary
        from index_mapping import load_mapping

        mapping = load_mapping('./root_id_to_index_mapping.json')
        rootid_mapping = dict((v, k) for k, v in mapping.items())

        #  build a cluster assignment dictionary
        cluster_assignment_dict = dict()
        for i in range(len(labels)):
            root_id = mapping[i]
            cluster_assignment_dict[root_id] = labels[i].item()

        # Save the results
        torch.save(U_orth.cpu(), 'U_orth.pt')
        print("Saved U and U_orth to disk.")

        # Save labels and model parameters as desired
        torch.save(labels.cpu(), 'mog_labels.pt')
        torch.save(learned_weights.detach().cpu(), 'mog_weights.pt')
        torch.save(learned_means.detach().cpu(), 'mog_means.pt')
        torch.save(learned_vars.detach().cpu(), 'mog_variances.pt')
        
        # Optionally, also save the responsibilities if needed
        torch.save(resp.cpu(), 'mog_responsibilities.pt')

        np.save("mog_cluster_assignment_dict.npy", cluster_assignment_dict, allow_pickle=True)
