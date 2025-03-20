import torch
from torch import nn
from scipy.sparse import csr_matrix, load_npz
import numpy as np
from tqdm import tqdm
import gc

from datetime import datetime

device="cpu"
if torch.backends.mps.is_available():
    device="mps"
if torch.cuda.is_available():
    device="cuda"

# set the global seed using torch. use the current time to make it more random.
torch.manual_seed(int(datetime.now().timestamp()))

# constants
k = 1024      # number of clusters: 729
d = 32       # dimensionality of feature space
minibatch_size = 2_500
num_m_updates = 10_000
n_epochs = 1_000

dtype=torch.float32
if device == "cuda":
    dtype=torch.bfloat16

logk = torch.log(torch.tensor(k, dtype=dtype))

# load the sparse matrix from `sparse_connectivity_matrix.npz`.
# the sparse matrix is in the format of csr_matrix.
adj_matrix = load_npz("./sparse_connectivity_matrix.npz")

# threshold the sparse matrix to obtain a binary adjacency matrix
adj_matrix.data = (adj_matrix.data > 0).astype(np.int8)

n = adj_matrix.shape[0]  # number of nodes
print(f"loaded adjacency matrix with shape {adj_matrix.shape} and {adj_matrix.nnz} non-zero entries.")

# prepare the incoming and outgoing edge stats
outgoing_sum = torch.tensor(adj_matrix.sum(axis=1), dtype=dtype).to(device).squeeze() / n
outgoing_sum_mean = outgoing_sum.mean()
incoming_sum = torch.tensor(adj_matrix.sum(axis=0), dtype=dtype).to(device).squeeze() / n
incoming_sum_mean = incoming_sum.mean()

# model parameters
u_left = nn.Parameter(1/np.sqrt(k * d) * torch.randn(k, d, dtype=dtype).to(device))
u_right = nn.Parameter(1/np.sqrt(k * d) * torch.randn(k, d, dtype=dtype).to(device))
bias = nn.Parameter(torch.log(torch.tensor([adj_matrix.mean()], dtype=dtype).to(device)))

# approximate posterior: must be sampled from a dirichlet distribution
q_logits = nn.Parameter(1/k * torch.randn(n, k, dtype=dtype).to(device))

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

# m-step: update u_left and u_right
def m_step(n_max_updates=None, optimizer=None):
    if n_max_updates is None:
        n_max_updates = 1
    if optimizer is None:
        optimizer = torch.optim.AdamW([U_left, U_right, bias, q_logits], lr=0.001)

    initial_loss = 0.
    final_loss = 0.

    # create a random permutation of the indices
    indices_i = torch.randperm(N, device=device)
    indices_j = torch.randperm(N, device=device)
    n_minibatches = N // minibatch_size

    for i in tqdm(range(n_minibatches)):

        if i > n_max_updates:
            print(f"Reached maximum number of updates {n_max_updates}.")
            break

        # Get the minibatch indices
        minibatch_indices_i = indices_i[i * minibatch_size: (i + 1) * minibatch_size]
        minibatch_indices_j = indices_j[i * minibatch_size: (i + 1) * minibatch_size]

        # Get the minibatch of q_logits
        q_probs_batch_i = torch.softmax(q_logits[minibatch_indices_i], dim=-1) # mb_size x K 
        q_probs_batch_j = torch.softmax(q_logits[minibatch_indices_j], dim=-1) # mb_size x K
        
        eps = 1e-10
        q_probs_batch_i = torch.clamp(q_probs_batch_i, eps, 1 - eps)
        q_probs_batch_j = torch.clamp(q_probs_batch_j, eps, 1 - eps)

        # get the sub-matrix of adjacency matrix
        CC = torch.tensor(adj_matrix[minibatch_indices_i.cpu().numpy()]
                          [:, minibatch_indices_j.cpu().numpy()].toarray(), 
                          dtype=dtype).to(device) # mb_size x mb_size

        # outer product of q_logits_batch_i and q_logits_batch_j
        edge_weighted_KK = torch.mm(q_probs_batch_i.t(), torch.mm(CC, q_probs_batch_j)) # K x K
        non_edge_weighted_KK = torch.mm(q_probs_batch_i.t(), torch.mm(1-CC, q_probs_batch_j)) # K x K
        
        # compute the edge probabilities
        e_prob = sigmoid(dot(U_left, U_right) + bias) # K x K
        e_prob = torch.clamp(e_prob, eps, 1 - eps)

        # compute the objective function
        obj = (edge_weighted_KK * log_(e_prob)).sum() + (non_edge_weighted_KK * log_(1 - e_prob)).sum()
        
        if i == n_minibatches-1:
            print("minibatch-i", minibatch_indices_i)
            print("minibatch-j", minibatch_indices_j)
            print("q-logits", q_logits)
            print("q-probs-i", q_probs_batch_i)
            print("q-probs-j", q_probs_batch_j)
            print("sub-matrix", CC)
            print("edge-weighted", edge_weighted_KK)
            print("non-edge-weighted", non_edge_weighted_KK)
            print("e-prob", e_prob)
            print("objective", obj)

        # add the entropy penalty on q_probs
        obj = obj - (q_probs_batch_i * log_(q_probs_batch_i)).sum(1).mean() / 2 
        obj = obj - (q_probs_batch_j * log_(q_probs_batch_j)).sum(1).mean() / 2

        # compute the loss
        loss = -obj

        # Zero the gradients
        optimizer.zero_grad()
        loss.backward()

        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_([U_left, U_right, bias, q_logits], 1.0)

        # import ipdb; ipdb.set_trace()
        # Perform a gradient step
        optimizer.step()
    
    print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")


# EM algorithm
optimizer = torch.optim.Adam([U_left, U_right, bias, q_logits], lr=1e-1)

try:
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        # M-step: Update U_left and U_right
        m_step(n_max_updates=num_m_updates, optimizer=optimizer)
        print("Cluster assignments:")
        for i in range(10):
            print(f"Node {i}: Cluster {torch.argmax(q_logits[i]).item()}")

except KeyboardInterrupt:
    print("Training interrupted.")

# Save results
cluster_assignments = torch.argmax(q_logits, dim=-1).cpu().numpy()  # Inferred cluster for each node
cluster_scores = torch.max(q_logits, dim=-1).values.cpu().detach().numpy()  # Confidence scores for each cluster
U_left_final = U_left.detach().cpu().numpy()
U_right_final = U_right.detach().cpu().numpy()

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
