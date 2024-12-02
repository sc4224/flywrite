import torch
from torch import nn
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from tqdm import tqdm
import gc

device="cpu"
# if torch.backends.mps.is_available():
#     device="mps"
if torch.cuda.is_available():
    device="cuda"

# Constants
K = 729      # Number of clusters
d = 32       # Dimensionality of feature space
batch_size = 2048  # Size of minibatch
num_epochs = 2_000
num_e_updates = 32
num_m_updates = 32

dtype=torch.float32
if device == "cuda":
    dtype=torch.bfloat16

# Load the sparse matrix from `sparse_connectivity_matrix.npz`.
# The sparse matrix is in the format of csr_matrix.
adj_matrix = load_npz("./sparse_connectivity_matrix.npz")

# Threshold the sparse matrix to obtain a binary adjacency matrix
adj_matrix.data = (adj_matrix.data > 0).astype(np.int8)

N = adj_matrix.shape[0]  # Number of nodes
print(f"Loaded adjacency matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

# Model parameters
U_left = nn.Parameter(0.0001 * torch.randn(K, d, dtype=dtype).to(device))
U_right = nn.Parameter(0.0001 * torch.randn(K, d, dtype=dtype).to(device))

# Variational parameters for Z (initialize uniform)
log_q_probs = 0.0001 * torch.randn(N, K, dtype=dtype).to(device)
moving_coeff = 0.  # Moving average coefficient for q_probs

def sigmoid(x, clamp=True):
    if clamp:
        x = torch.clamp(x, min=-5, max=5)
    return 1./(1+ torch.exp(-x))

def compute_dense_submatrix(indices, rows_only=False):
    """
    Given a set of indices, extract a dense submatrix from the sparse adjacency matrix.
    """
    if rows_only:
        sub_adj = adj_matrix[indices, :].toarray()
    else:
        sub_adj = adj_matrix[indices, :][:, indices].toarray()
    return torch.tensor(sub_adj, dtype=dtype).to(device)

# E-step: Update q_probs explicitly using dense submatrices
def compute_q_probs(n_max_updates=None):
    """
    Compute q_probs using dense submatrices for tractability, with shuffled node indices.
    Optimized using batched matrix operations.
    """
    global log_q_probs
    log_q = torch.zeros(N, K, dtype=dtype).to(device)

    # Shuffle indices to mix different vertices
    perm = torch.randperm(N)

    # Compute logits for all cluster assignments (z_i, z_j)
    logits = torch.mm(U_left, U_right.t())  # Shape: (K, K)
    logits = logits.unsqueeze(2).unsqueeze(3)  # Shape: (K, K, 1, 1)

    # Compute probabilities for all edges and non-edges
    prob_edges = sigmoid(logits)  # Shape: (K, K, 1, 1)
    log_prob_edges = torch.log(prob_edges + 1e-5)  # Shape: (K, K, 1, 1)
    log_prob_no_edges = torch.log(1 - prob_edges + 1e-5)  # Shape: (K, K, 1, 1)

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
        dense_submatrix = compute_dense_submatrix(batch_indices)  # Shape: (batch_size, N)

       # Compute D_i: Sum over neighbors for each node in the batch
        D_i = dense_submatrix.sum(dim=1)  # Shape: (batch_size,)
        D_i_comp = N - D_i  # Complement of D_i

        # Squeeze the probability tensors to remove unnecessary dimensions
        s_edges_left = log_prob_edges.sum(dim=1).squeeze(-1).squeeze(-1)  # Shape: (K,)
        s_no_edges_left = log_prob_no_edges.sum(dim=1).squeeze(-1).squeeze(-1)  # Shape: (K,)

        s_edges_right = log_prob_edges.sum(dim=0).squeeze(-1).squeeze(-1)  # Shape: (K,)
        s_no_edges_right = log_prob_no_edges.sum(dim=0).squeeze(-1).squeeze(-1)  # Shape: (K,)

        # Compute the log likelihood contributions without creating the large tensor
        log_likelihood_left = torch.outer(s_edges_left, D_i) + torch.outer(s_no_edges_left, D_i_comp)  # Shape: (K, batch_size)
        log_likelihood_right = torch.outer(s_edges_right, D_i) + torch.outer(s_no_edges_right, D_i_comp)  # Shape: (K, batch_size)

        # Sum the contributions to get the final log likelihood per node
        log_likelihood_node = (log_likelihood_left + log_likelihood_right) / batch_size  # Shape: (K, batch_size)

        # Subtract the maximum value for numerical stability
        log_likelihood_node = log_likelihood_node - log_likelihood_node.max(dim=0).values

        # Update posterior probabilities for this batch
        log_q_probs[batch_indices] = (moving_coeff * log_q_probs[batch_indices] + \
            (1 - moving_coeff) * torch.log(torch.softmax(log_likelihood_node.t(), dim=-1)+1e-5)).detach()

        if torch.isnan(log_q_probs[batch_indices]).any():
            print("NaN detected in log_q_probs[batch_indices]. Skipping update.")
            import ipdb; ipdb.set_trace()


# M-step: Update U_left and U_right
def m_step(n_max_updates=None):
    optimizer = torch.optim.Adam([U_left, U_right], lr=0.001)
    perm = torch.randperm(N)

    # Initialize loss tracking
    initial_loss = 0.0
    final_loss = 0.0

    print("M-Step: Updating U_left and U_right...")
    bi = 0
    for batch_start in tqdm(range(0, len(perm), batch_size)):
        bi = bi + 1

        if n_max_updates is not None and bi > n_max_updates:
            break

        batch_end = min(batch_start + batch_size, len(perm))
        batch_indices = perm[batch_start:batch_end]

        # Extract dense submatrix for this batch
        dense_submatrix = compute_dense_submatrix(batch_indices)

        # Compute the probability distribution over Z
        log_q_probs_batch = log_q_probs[batch_indices].detach()
        log_q_probs_batch = log_q_probs_batch - log_q_probs_batch.max(dim=-1).values.unsqueeze(-1)
        P_Z = torch.softmax(log_q_probs_batch, dim=-1)  # Shape: [batch_size, num_classes]

        # Compute the inner products between all pairs of U_left and U_right embeddings
        scores = U_left @ U_right.T  # Shape: [num_classes, num_classes]

        # Compute the sigmoid of the clamped scores
        sigmoid_scores = sigmoid(scores)  # Shape: [num_classes, num_classes]

        if torch.isnan(scores).any():
            print("NaN detected in scores. Skipping update.")
            import ipdb; ipdb.set_trace()

        # Compute the expected edge probabilities over Z
        edge_probs = P_Z @ sigmoid_scores @ P_Z.T  # Shape: [batch_size, batch_size]

        if torch.isnan(edge_probs).any():
            print("NaN detected in edge_probs. Skipping update.")
            import ipdb; ipdb.set_trace()

        # Define the label smoothing parameter
        smoothing = 0.1  # Adjust this value as needed (typically small, e.g., 0.1)

        # Apply label smoothing to the target labels
        smoothed_labels = dense_submatrix * (1 - smoothing) + smoothing * 0.5

        # Loss for this minibatch with label smoothing
        loss = nn.BCELoss()(edge_probs, smoothed_labels)

        # Update loss tracking
        if batch_start == 0:
            initial_loss = loss.item()  # Record the initial loss
        final_loss = loss.item()  # Update the final loss at the end of each batch

        # Update embeddings
        optimizer.zero_grad()
        loss.backward()

        if torch.isnan(U_left.grad).any() or torch.isnan(U_right.grad).any():
            print("NaN detected in gradients. Skipping update.")
            import ipdb; ipdb.set_trace()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(U_left, 1.0)
        torch.nn.utils.clip_grad_norm_(U_right, 1.0)

        optimizer.step()

    print(f"M-step: Initial Loss = {initial_loss:.4f}, Final Loss = {final_loss:.4f}")

# EM algorithm
try:
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # E-step: Update q_probs explicitly using dense submatrices
        compute_q_probs(n_max_updates=num_e_updates)
        print(f"Updated q_probs for E-step (sample): {torch.argmax(log_q_probs[:10], dim=-1).cpu().numpy()}")
        gc.collect()

        # M-step: Update U_left and U_right
        m_step(n_max_updates=num_m_updates)
        gc.collect()
except KeyboardInterrupt:
    print("Training interrupted.")

# Save results
cluster_assignments = torch.argmax(log_q_probs, dim=-1).cpu().numpy()  # Inferred cluster for each node
cluster_scores = torch.max(log_q_probs, dim=-1).values.cpu().detach().numpy()  # Confidence scores for each cluster
U_left_final = U_left.detach().cpu().numpy()
U_right_final = U_right.detach().cpu().numpy()

# build a index-to-root_id dictionary
from index_mapping import load_mapping, matrix_index_to_root_id

mapping = load_mapping('./root_id_to_index_mapping.json')
rootid_mapping = dict((v, k) for k, v in mapping.items())

# build a cluster assignment dictionary
cluster_assignment_dict = dict()
for i in range(len(cluster_assignments)):
    root_id = matrix_index_to_root_id(i, rootid_mapping)
    cluster_assignment_dict[root_id] = cluster_assignments[i]

np.save("cluster_assignments.npy", cluster_assignments)
np.save("cluster_scores.npy", cluster_scores)
np.save("U_left.npy", U_left_final)
np.save("U_right.npy", U_right_final)
np.save("cluster_assignment_dict.npy", cluster_assignment_dict)

print("Results saved: 'cluster_assignments.npy', 'U_left.npy', 'U_right.npy', 'cluster_assignment_dict.npy'")