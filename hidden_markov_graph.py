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
K = 729      # Number of clusters: 729
d = 32       # Dimensionality of feature space
num_m_updates = 1_000

dtype=torch.float32
if device == "cuda":
    dtype=torch.bfloat16

logK = torch.log(torch.tensor(K, dtype=dtype))

# Load the sparse matrix from `sparse_connectivity_matrix.npz`.
# The sparse matrix is in the format of csr_matrix.
adj_matrix = load_npz("./sparse_connectivity_matrix.npz")

# Threshold the sparse matrix to obtain a binary adjacency matrix
adj_matrix.data = (adj_matrix.data > 0).astype(np.int8)

N = adj_matrix.shape[0]  # Number of nodes
print(f"Loaded adjacency matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

# prepare the incoming and outgoing edge stats
outgoing_sum = torch.tensor(adj_matrix.sum(axis=1), dtype=dtype).squeeze() / N
outgoing_sum_mean = outgoing_sum.mean()
incoming_sum = torch.tensor(adj_matrix.sum(axis=0), dtype=dtype).squeeze() / N
incoming_sum_mean = incoming_sum.mean()

# Model parameters
U_left = nn.Parameter(1e-1 * torch.randn(K, d, dtype=dtype).to(device))
U_right = nn.Parameter(1e-1 * torch.randn(K, d, dtype=dtype).to(device))
bias = nn.Parameter(torch.log(torch.tensor([adj_matrix.mean()], dtype=dtype).to(device)))

# approximate posterior: must be sampled from a Dirichlet distribution
q_logits = nn.Parameter(1/np.sqrt(K) * torch.randn(N, K, dtype=dtype).to(device))

def sigmoid(x, clamp=False):
    if clamp:
        x = torch.clamp(x, min=-5, max=5)
    return 1./(1+ torch.exp(-x))

def log_(x):
    return torch.log(torch.clamp(x, min=1e-6))

def dot(x, y):
    return torch.mm(x, y.t())

def cosine(x, y):
    return torch.mm(x, y.t()) / (torch.norm(x, dim=-1) * torch.norm(y, dim=-1))

# M-step: Update U_left and U_right
def m_step(n_max_updates=None, optimizer=None):
    if n_max_updates is None:
        n_max_updates = 1
    if optimizer is None:
        optimizer = torch.optim.AdamW([U_left, U_right, bias, q_logits], lr=0.001)

    initial_loss = 0.
    final_loss = 0.

    for i in tqdm(range(n_max_updates)):
        optimizer.zero_grad()

        # Compute the gradient of the ELBO w.r.t. U_left and U_right
        CC = sigmoid(dot(U_left, U_right) + bias)

        logc_sum = log_(CC).sum(dim=1)
        log1c_sum = log_(1-CC).sum(dim=1)
        outgoing = torch.outer(outgoing_sum, logc_sum)
        outgoing = outgoing + outgoing_sum_mean * torch.outer(1 - outgoing_sum, log1c_sum)

        logc_sum = log_(CC).sum(dim=0)
        log1c_sum = log_(1-CC).sum(dim=0)
        incoming = torch.outer(incoming_sum, logc_sum) 
        incoming = incoming + incoming_sum_mean * torch.outer(1 - incoming_sum, log1c_sum)

        overall = incoming + outgoing
        
        # weighted sum of `overall` using q_probs
        overall = (torch.softmax(q_logits, dim=-1) * overall).sum(dim=-1)
        loss = -overall.sum()

        loss.backward()

        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_([U_left, U_right, bias, q_logits], 1.0)

        # Perform a gradient step
        optimizer.step()

        if i % 10 == 0:
            print(f"Loss at iteration {i}: {loss.item()}")
            print("Cluster assignments:")
            for i in range(10):
                print(f"Node {i}: Cluster {torch.argmax(q_logits[i]).item()}")

    # import ipdb; ipdb.set_trace()

    print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")


# EM algorithm
optimizer = torch.optim.AdamW([U_left, U_right, bias, q_logits], lr=0.001)

try:
    # M-step: Update U_left and U_right
    m_step(n_max_updates=num_m_updates, optimizer=optimizer)
except KeyboardInterrupt:
    print("Training interrupted.")

# Save results
cluster_assignments = torch.argmax(q_logits, dim=-1).cpu().numpy()  # Inferred cluster for each node
cluster_scores = torch.max(q_logits, dim=-1).values.cpu().detach().numpy()  # Confidence scores for each cluster
U_left_final = U_left.detach().cpu().numpy()
U_right_final = U_right.detach().cpu().numpy()
# U_left_final = U_left_mu.detach().cpu().numpy()
# U_right_final = U_right_mu.detach().cpu().numpy()

# build a index-to-root_id dictionary
from index_mapping import load_mapping, matrix_index_to_root_id

mapping = load_mapping('./root_id_to_index_mapping.json')
rootid_mapping = dict((v, k) for k, v in mapping.items())

# build a cluster assignment dictionary
cluster_assignment_dict = dict()
for i in range(len(cluster_assignments)):
    root_id = mapping[i]
    cluster_assignment_dict[root_id] = cluster_assignments[i].item()

np.save("cluster_assignments.npy", cluster_assignments)
np.save("cluster_scores.npy", cluster_scores)
np.save("U_left.npy", U_left_final)
np.save("U_right.npy", U_right_final)
np.save("cluster_assignment_dict.npy", cluster_assignment_dict)

print("Results saved: 'cluster_assignments.npy', 'U_left.npy', 'U_right.npy', 'cluster_assignment_dict.npy'")