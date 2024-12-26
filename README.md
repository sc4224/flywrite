# Clustering of Neurons from the Fruit Fly Connectome

## Download data

i've used the following data (unfiltered version) to create a connectivity matrix:

* https://codex.flywire.ai/api/download?data_product=connections_no_threshold&data_version=783

i've used the following file to map a root id to its name:

* https://codex.flywire.ai/api/download?data_product=names&data_version=783

## Steps

```
> pip install -r ./requirements.txt
> python ./connectivity_matrix_construction.py
> python ./visual_neuron_type_dict.py
> python ./hidden_markov_graph.py
> python ./sparse_graph_pca.py
```

## For analysis

use `cluster_similarity_test.ipynb` with jupyter notebook.

## Write up

see [<Stochastic variational inference for low-rank stochastic block models, or how i re-discovered SBM unnecessarily>](https://kyunghyuncho.me/stochastic-variational-inference-for-low-rank-stochastic-block-models-or-how-i-re-discovered-sbm-unnecessarily/) for more details.

## Commit history

don't bother with the commit history, as this history is more or less how i was misled constantly over several days if not weeks by ChatGPT and did not realized i was being misled.

