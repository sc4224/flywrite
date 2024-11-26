import pyro
import pyro.distributions as dist
import torch
import numpy as np

from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from scipy.sparse import load_npz
from tqdm import tqdm

# Constants
K = 256      # Number of clusters
d = 10       # Dimensionality of feature space

# load the sparse matrix from `sparse_connectivity_matrix.npz`.
# the sparse matrix is in the format of csr_matrix.
sparse_matrix = load_npz("sparse_connectivity_matrix.npz")

N = sparse_matrix.shape[0]  # Number of nodes

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Model definition
def model(observed_sparse_matrix):
    # Latent cluster assignments
    Z = pyro.sample("Z", dist.Categorical(logits=torch.zeros(N, K)))
    
    # Feature matrix for each cluster
    U = pyro.param("U", torch.randn(K, d), constraint=constraints.real)
    
    # Iterate over all pairs of nodes
    for i in range(observed_sparse_matrix.shape[0]):
        row_start = observed_sparse_matrix.indptr[i]
        row_end = observed_sparse_matrix.indptr[i + 1]
        non_zero_columns = set(observed_sparse_matrix.indices[row_start:row_end])
        
        for j in range(N):
            if j in non_zero_columns:
                obs = observed_sparse_matrix[i, j]
            else:
                obs = 0  # Implied no edge
            
            U_i = U[Z[i]]
            U_j = U[Z[j]]
            mu_ij = sigmoid((U_i * U_j).sum())
            pyro.sample(f"W_{i}_{j}", dist.Bernoulli(mu_ij), obs=torch.tensor(obs, dtype=torch.float32))

# Guide definition
def guide(observed_sparse_matrix):
    # Learnable parameters for variational distribution over Z
    q_logits = pyro.param("q_logits", torch.randn(N, K))
    pyro.sample("Z", dist.Categorical(logits=q_logits))

# Use csr_matrix for observed data
observed_matrix = sparse_matrix

# Set up the optimizer and inference
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Training loop with a progress bar
num_steps = 1000
with tqdm(total=num_steps, desc="Training Progress", unit="step") as pbar:
    for step in range(num_steps):
        loss = svi.step(observed_matrix)
        if step % 100 == 0:
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
        pbar.update(1)

# Output the learned cluster assignments
q_logits = pyro.param("q_logits")
Z_posterior = torch.argmax(q_logits, dim=1)

# Save cluster assignments to a file
output_file = "cluster_assignments.npy"
np.save(output_file, Z_posterior.numpy())
print(f"Inferred cluster assignments saved to {output_file}.")