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
logK = torch.log(torch.tensor(K))
d = 64       # Dimensionality of feature space
num_epochs = 2_000
num_m_updates = 10

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

# prepare the incoming and outgoing edge stats
outgoing_sum = torch.tensor(adj_matrix.sum(axis=1)).squeeze() / N
outgoing_sum_mean = outgoing_sum.mean()
incoming_sum = torch.tensor(adj_matrix.sum(axis=0)).squeeze() / N
incoming_sum_mean = incoming_sum.mean()

# Model parameters
U_left_mu = nn.Parameter(1e-2 * torch.randn(K, d, dtype=dtype).to(device))
U_right_mu = nn.Parameter(1e-2 * torch.randn(K, d, dtype=dtype).to(device))
U_left_logs = nn.Parameter(1e-2 * torch.randn(K, d, dtype=dtype).to(device))
U_right_logs = nn.Parameter(1e-2 * torch.randn(K, d, dtype=dtype).to(device))

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

# E-step: Update q_probs explicitly using dense submatrices
def compute_q_probs(q_probs_old=None):
    CC = sigmoid(dot(U_left_mu, U_right_mu)).detach() # (K, K)

    logc_sum = log_(CC).sum(dim=1)
    log1c_sum = log_(1-CC).sum(dim=1)
    outgoing = torch.outer(outgoing_sum, logc_sum) + outgoing_sum_mean * torch.outer(1 - outgoing_sum, log1c_sum)

    logc_sum = log_(CC).sum(dim=0)
    log1c_sum = log_(1-CC).sum(dim=0)
    incoming = torch.outer(incoming_sum, logc_sum) + incoming_sum_mean * torch.outer(1 - incoming_sum, log1c_sum)

    overall = incoming + outgoing

    if q_probs_old is None:
        q_probs = torch.softmax(overall, dim=-1)
    else:
        q_probs = 0.9 * q_probs_old + 0.1 * torch.softmax(overall, dim=-1)

    if torch.isnan(q_probs).any():
        print("NaNs detected in q_probs!")
        import ipdb; ipdb.set_trace()

    return q_probs.detach()

# M-step: Update U_left and U_right
def m_step(q_probs, n_max_updates=None, optimizer=None):
    if n_max_updates is None:
        n_max_updates = 1
    if optimizer is None:
        optimizer = torch.optim.Adam([U_left_mu, U_right_mu, U_left_logs, U_right_logs], lr=0.001)

    initial_loss = 0.
    final_loss = 0.

    for i in tqdm(range(n_max_updates)):
        optimizer.zero_grad()

        # Compute the gradient of the ELBO w.r.t. U_left and U_right
        U_left = U_left_mu + torch.exp(U_left_logs) * torch.randn_like(U_left_mu)
        U_right = U_right_mu + torch.exp(U_right_logs) * torch.randn_like(U_right_mu)
        CC = sigmoid(dot(U_left, U_right))

        logc_sum = log_(CC).sum(dim=1)
        log1c_sum = log_(1-CC).sum(dim=1)
        outgoing = torch.outer(outgoing_sum, logc_sum) + outgoing_sum_mean * torch.outer(1 - outgoing_sum, log1c_sum)

        logc_sum = log_(CC).sum(dim=0)
        log1c_sum = log_(1-CC).sum(dim=0)
        incoming = torch.outer(incoming_sum, logc_sum) + incoming_sum_mean * torch.outer(1 - incoming_sum, log1c_sum)

        overall = incoming + outgoing + log_(q_probs.detach())

        loss = -torch.logsumexp(overall, dim=-1).mean()

        # KL divergence between Normal(U_left_mu, U_left_logs) and Normal(0, 1)
        kl_left = 0.5 * (U_left_mu.pow(2) + U_left_logs.exp() - 1 - U_left_logs).sum()
        # KL divergence between Normal(U_right_mu, U_right_logs) and Normal(0, 1)
        kl_right = 0.5 * (U_right_mu.pow(2) + U_right_logs.exp() - 1 - U_right_logs).sum()

        loss += 1. * (kl_left + kl_right)
        loss.backward()

        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(U_left_mu, 1.0)
        torch.nn.utils.clip_grad_norm_(U_right_mu, 1.0)
        torch.nn.utils.clip_grad_norm_(U_left_logs, 1.0)
        torch.nn.utils.clip_grad_norm_(U_right_logs, 1.0)

        # Perform a gradient step
        optimizer.step()

    # import ipdb; ipdb.set_trace()

    print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")


# EM algorithm
optimizer = torch.optim.Adam([U_left_mu, U_right_mu, U_left_logs, U_right_logs], lr=0.001)

q_probs = None

try:
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # E-step: Update q_probs explicitly using dense submatrices
        q_probs = compute_q_probs(q_probs_old=q_probs)
        # now print the cluster assignments of the first 10 items.
        # make sure to print their log-probabilities (up to 3 digits) as well.
        print("Cluster assignments:")
        for i in range(10):
            print(f"Node {i}: Cluster {torch.argmax(q_probs[i]).item()} (Prob: {q_probs[i].max().item():.3f})")

        gc.collect()

        # M-step: Update U_left and U_right
        m_step(q_probs, n_max_updates=num_m_updates, optimizer=optimizer)
        gc.collect()
except KeyboardInterrupt:
    print("Training interrupted.")

# Save results
cluster_assignments = torch.argmax(q_probs, dim=-1).cpu().numpy()  # Inferred cluster for each node
cluster_scores = torch.max(q_probs, dim=-1).values.cpu().detach().numpy()  # Confidence scores for each cluster
U_left_final = U_left_mu.detach().cpu().numpy()
U_right_final = U_right_mu.detach().cpu().numpy()

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