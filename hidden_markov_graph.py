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
batch_size = 500  # Size of minibatch

# Load the sparse matrix from `sparse_connectivity_matrix.npz`.
# The sparse matrix is in the format of csr_matrix.
sparse_matrix = load_npz("sparse_connectivity_matrix.npz")

# Threshold the sparse matrix to obtain a binary adjacency matrix
sparse_matrix.data = (sparse_matrix.data > 0).astype(np.int8)

N = sparse_matrix.shape[0]  # Number of nodes

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Minibatch sampler
def sample_minibatch(sparse_matrix, batch_size):
    sampled_indices = np.random.choice(N, size=batch_size, replace=False)
    submatrix = sparse_matrix[sampled_indices, :][:, sampled_indices].toarray()
    return sampled_indices, torch.tensor(submatrix, dtype=torch.float32)

# Model definition
def model(sampled_indices, observed_submatrix):
    # Plate for sampled nodes
    with pyro.plate("nodes", len(sampled_indices)):
        # Latent cluster assignments
        Z = pyro.sample("Z", dist.Categorical(logits=torch.zeros(len(sampled_indices), K)))
    
    # Feature matrix for each cluster
    with pyro.plate("clusters", K):
        U = pyro.sample("U", dist.Normal(torch.zeros(d), torch.ones(d)).to_event(1))
    
    # Plate for pairwise interactions within the minibatch
    num_pairs = len(sampled_indices) ** 2
    with pyro.plate("pairs", num_pairs):
        # Compute pairwise indices
        pairwise_indices = torch.cartesian_prod(torch.arange(len(sampled_indices)), torch.arange(len(sampled_indices)))
        i_indices, j_indices = pairwise_indices[:, 0], pairwise_indices[:, 1]
        
        # Compute cluster embeddings
        U_i = U[Z[i_indices]]
        U_j = U[Z[j_indices]]
        
        # Pairwise interaction
        pairwise_interaction = torch.einsum("bd,bd->b", U_i, U_j)  # Dot product along feature dimension
        mu_ij = sigmoid(pairwise_interaction)
        
        # Flatten observed_submatrix and match to pairwise interactions
        observed_flat = observed_submatrix.flatten()
        pyro.sample("W", dist.Bernoulli(mu_ij), obs=observed_flat)

# Guide definition
def guide(sampled_indices, observed_submatrix):
    # Plate for sampled nodes
    with pyro.plate("nodes", len(sampled_indices)):
        # Learnable parameters for variational distribution over Z
        q_logits = pyro.param("q_logits", torch.randn(N, K))
        Z = pyro.sample("Z", dist.Categorical(logits=q_logits[sampled_indices]))
    
    # Plate for clusters
    with pyro.plate("clusters", K):
        # Variational distribution over U
        q_mu = pyro.param("q_mu", torch.randn(K, d))
        q_sigma = pyro.param("q_sigma", torch.ones(K, d), constraint=constraints.positive)
        U = pyro.sample("U", dist.Normal(q_mu, q_sigma).to_event(1))

# Set up the optimizer and inference
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Training loop with minibatching
num_steps = 1000
with tqdm(total=num_steps, desc="Training Progress", unit="step") as pbar:
    for step in range(num_steps):
        # Sample a minibatch
        sampled_indices, submatrix = sample_minibatch(sparse_matrix, batch_size)
        
        # Perform one step of inference
        loss = svi.step(sampled_indices, submatrix)
        if step % 1 == 0:
            pbar.set_postfix({"Loss": f"{loss:.4f}"})
        pbar.update(1)

# Output the learned cluster assignments
q_logits = pyro.param("q_logits")
Z_posterior = torch.argmax(q_logits, dim=1)

# Output the learned U
q_mu = pyro.param("q_mu")
q_sigma = pyro.param("q_sigma")
U_posterior = q_mu  # Posterior mean as the point estimate

# Save results to files
np.save("cluster_assignments.npy", Z_posterior.numpy())
np.save("U_posterior_mean.npy", U_posterior.numpy())
print("Inferred cluster assignments and U posterior mean saved to files.")