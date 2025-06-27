import json
import os
import time
import numpy as np
from skopt import Optimizer
from skopt.space import Integer, Real, Categorical
from skopt.utils import point_asdict

space_list = [
    Integer(256, 2048, prior='log-uniform', name='k'),
    Integer(16, 48, name='d'),
    Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
    Categorical(['Adam', 'AdamW', 'SGD'], name='optimizer')
]

# Convert to dictionary for point_asdict
space_dict = {dim.name: dim for dim in space_list}

batch_size = 50
n_batches = 1

def wait_for_results(ids):
    while True:
        if all(os.path.exists(f"results/result_{i}.json") for i in ids):
            break
        time.sleep(10)

def load_results(ids):
    scores = []
    for i in ids:
        with open(f"results/result_{i}.json") as f:
            result = json.load(f)
            scores.append((result["elbo"], result["best_epoch"]))
    return scores

def sanitize_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

if __name__ == "__main__":
   os.system(f"sbatch --array=0-{batch_size-1} run_sbm.sh")
   #  opt = Optimizer(dimensions=space_list, base_estimator="GP", acq_func="EI", random_state=42)

   #  for batch_idx in range(n_batches):
   #      candidates = opt.ask(n_points=batch_size)

   #      # Save candidates
   #      os.makedirs("configs", exist_ok=True)
   #      os.makedirs("results", exist_ok=True)

   #      for i, params in enumerate(candidates):
   #          params_dict = point_asdict(space_dict, params)
   #          # Sanitize numpy types for JSON serialization
   #          sanitized_params = {k: sanitize_json(v) for k, v in params_dict.items()}
   #          with open(f"configs/params_{i}.json", "w") as f:
   #              json.dump(sanitized_params, f)

   #      # Submit SLURM array job
   #      os.system(f"sbatch --array=0-{batch_size-1} run_sbm.sh")

   #      # Wait for results
   #      wait_for_results(range(batch_size))

   #      # Collect and use results
   #      scores = load_results(range(batch_size))
   #      lowest_elbos, best_epochs = zip(*scores)

   #      opt.tell(candidates, lowest_elbos)
   #      print(f"Batch {batch_idx+1}: Best score so far = {min(opt.yi)}")

    # best_idx = np.argmin(opt.yi)
    # print("\nBest configuration:")
    # print(f"  Params: {opt.Xi[best_idx]}")
    # print(f"  Best Epoch: {best_epochs[best_idx]}")
    # print(f"  Best Validation ELBO: {opt.yi[best_idx]}")

