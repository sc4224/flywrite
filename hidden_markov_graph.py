import torch
from torch import nn
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from tqdm import tqdm

# Constants
K = 8      # Number of clusters
d = 32       # Dimensionality of feature space
batch_size = 512  # Size of minibatch
num_epochs = 10

# Load the sparse matrix from `sparse_connectivity_matrix.npz`.
# The sparse matrix is in the format of csr_matrix.
adj_matrix = load_npz("sparse_connectivity_matrix.npz")

# Threshold the sparse matrix to obtain a binary adjacency matrix
adj_matrix.data = (adj_matrix.data > 0).astype(np.int8)

N = adj_matrix.shape[0]  # Number of nodes

# Model parameters
U_left = nn.Parameter(torch.randn(K, d))
U_right = nn.Parameter(torch.randn(K, d))

# Variational parameters for Z (initialize uniform)
q_probs = torch.ones(N, K) / K  # Posterior probabilities of Z
moving_coeff = 0.9  # Moving average coefficient for q_probs

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def compute_dense_submatrix(indices, rows_only=False):
    """
    Given a set of indices, extract a dense submatrix from the sparse adjacency matrix.
    """
    if rows_only:
        sub_adj = adj_matrix[indices, :].toarray()
    else:
        sub_adj = adj_matrix[indices, :][:, indices].toarray()
    return torch.tensor(sub_adj, dtype=torch.float32)

# E-step: Update q_probs explicitly using dense submatrices
def compute_q_probs(n_max_updates=None):
    """
    Compute q_probs using dense submatrices for tractability, with shuffled node indices.
    Optimized using batched matrix operations.
    """
    global q_probs
    log_q = torch.zeros(N, K)

    # Shuffle indices to mix different vertices
    perm = torch.randperm(N)

    # Compute logits for all cluster assignments (z_i, z_j)
    logits = torch.mm(U_left, U_right.t())  # Shape: (K, K)
    logits = logits.unsqueeze(2).unsqueeze(3)  # Shape: (K, K, 1, 1)

    # Compute probabilities for all edges and non-edges
    prob_edges = sigmoid(logits)  # Shape: (K, K, 1, 1)
    log_prob_edges = torch.log(prob_edges + 1e-9)  # Shape: (K, K, 1, 1)
    log_prob_no_edges = torch.log(1 - prob_edges + 1e-9)  # Shape: (K, K, 1, 1)

    # Process in minibatches
    print("E-Step: Computing q_probs using dense submatrices...")

    bi = 0
    for batch_start in tqdm(range(0, N, batch_size)):
        bi = bi + 1
        if n_max_updates is not None and bi > n_max_updates:
            break

        batch_end = min(batch_start + batch_size, N)
        batch_indices = perm[batch_start:batch_end]

        # Extract dense submatrix for this batch
        dense_submatrix = compute_dense_submatrix(batch_indices)  # Shape: (batch_size, batch_size)

        # Compute likelihood contributions for all nodes in the batch
        dense_submatrix = dense_submatrix.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, batch_size, batch_size)
        log_likelihood = (
            dense_submatrix * log_prob_edges
            + (1 - dense_submatrix) * log_prob_no_edges
        )  # Shape: (K, K, batch_size, N)

        # Sum over neighbors to get log probabilities for Z_i assignments
        log_likelihood_left = log_likelihood.sum(dim=-1).sum(dim=1) # (K, batch_size)
        log_likelihood_right = log_likelihood.sum(dim=-2).sum(dim=0) # (K, batch_size)
        log_likelihood_node = log_likelihood_left + log_likelihood_right  # (K, batch_size)

        # Update posterior probabilities for this batch
        q_probs[batch_indices] = moving_coeff * q_probs[batch_indices] + \
            (1 - moving_coeff) * torch.softmax(log_likelihood_node.t(), dim=-1)

# M-step: Update U_left and U_right
def m_step():
    optimizer = torch.optim.Adam([U_left, U_right], lr=0.01)
    perm = torch.randperm(N)

    # Initialize loss tracking
    initial_loss = 0.0
    final_loss = 0.0

    print("M-Step: Updating U_left and U_right...")
    for batch_start in tqdm(range(0, len(perm), batch_size)):
        batch_end = min(batch_start + batch_size, len(perm))
        batch_indices = perm[batch_start:batch_end]

        # Extract dense submatrix for this batch
        dense_submatrix = compute_dense_submatrix(batch_indices)

        # Compute likelihood for edges
        Z_samples = torch.multinomial(q_probs[batch_indices], num_samples=1).squeeze(-1)
        edge_probs = sigmoid(
            (U_left[Z_samples[:, None]] @ U_right[Z_samples[None, :]].transpose(1, 2)).squeeze()
        )

        # Loss for this minibatch
        loss = nn.BCELoss()(edge_probs, dense_submatrix)

        # Update loss tracking
        if batch_start == 0:
            initial_loss = loss.item()  # Record the initial loss
        final_loss = loss.item()  # Update the final loss at the end of each batch

        # Update embeddings
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"M-step: Initial Loss = {initial_loss:.4f}, Final Loss = {final_loss:.4f}")

# EM algorithm
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # E-step: Update q_probs explicitly using dense submatrices
    compute_q_probs(n_max_updates=5)
    print(f"Updated q_probs for E-step (sample): {q_probs[:5]}")

    # M-step: Update U_left and U_right
    m_step()

# Save results
cluster_assignments = torch.argmax(q_probs, dim=-1).cpu().numpy()  # Inferred cluster for each node
U_left_final = U_left.detach().cpu().numpy()
U_right_final = U_right.detach().cpu().numpy()

np.save("cluster_assignments.npy", cluster_assignments)
np.save("U_left.npy", U_left_final)
np.save("U_right.npy", U_right_final)

print("Results saved: 'cluster_assignments.npy', 'U_left.npy', 'U_right.npy'")