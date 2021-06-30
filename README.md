# Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training
This repository is an implementation of paper: [Scalable and Adaptive Graph Neural Networks with Self-Label-Enhanced training](https://arxiv.org/abs/2104.09376)

## requirements
`pytorch dgl ogb pytorch-geometric`

## datasets
Inductive datasets (Reddit, Flickr, Yelp) can be loaded via dgl or pyg. PPI should be separately downloaded and saved into `dataset`. More detailed information can be found in [GraphSaint](https://github.com/GraphSAINT/GraphSAINT)

OGB datasets are loaded via OGB library. More detailed information can be found in [Open Graph Benchmark](https://ogb.stanford.edu/)

## Reproduce results
Run shell scripts on folder `scripts`, add device number to indicate GPU:

`bash scripts/{dataset}/{method} {device_id}`

## Results
Test F1-micro with means and standard deviations under 10 runs. We use reported metrics for baselines.
| Method | Reddit | Flickr |  PPI | Yelp |
| ---- | ---- | ---- | ---- | ---- |
|GCN |93.3±0.0\% | 49.2±0.3\% | 51.5±0.6\% | 37.8±0.1\% |
|FastGCN | 92.4±0.1\% | 50.4±0.1\% | 51.3±3.2\% | 26.5±5.3\% |
|Stochastic-GCN | 96.4±0.1\% | 48.2±0.3\% | 96.3±1.0\% | 64.0±0.2\%|
|AS-GCN | 95.8±0.1\% | 50.4±0.2\% | 68.7±1.2\% | — |
|GraphSAGE | 95.3±0.1\% | 50.1±1.3\% | 63.7±0.6\% | 63.4±0.6\%|
|ClusterGCN | 95.4±0.1\% | 48.1±0.5\% | 87.5±0.4\% | 60.9±0.5\% |
|GraphSaint | 96.6±0.1\% | 51.1±0.1\% | **98.1±0.4\%** | **65.3±0.3\%**|
|SGC | 94.9±0.0\% | 50.2±0.1\% | 89.2±1.5\% | 35.8±0.6\%|
|SIGN | 96.8±0.0\% | 51.4±0.1\% | 97.0±0.3\% | 63.1±0.3\%|
|SAGN | 96.9±0.0\% | 51.4±1.2\% | 97.9±0.1\% | **65.3±0.1\%**|
|SAGN+1-SLE | **97.1±0.0\%** | 54.3±0.5\% | 98.0±0.1\% | **65.3±0.1\%**|
|SAGN+2-SLE | **97.1±0.0\%** | **54.6±0.4\%** | 98.0±0.1\% | **65.3±0.1\%**|

Validation and test accuray with means and standard deviations under 10 runs. We use reported metrics for baselines.
| Method | ogbn-products validation | ogbn-products test | ogbn-papers100M validation | ogbn-papers100M test|
| ---- | ---- | ---- | ----- | ----|
|MLP | 75.54±0.14\% | 61.06±0.08\% | 49.60±0.29\% | 47.24±0.31\%|
| Node2Vec | 90.32±0.06\% | 72.49±0.10\% | 58.07±0.28\% | 55.60±0.23\% |
| GCN | 92.00±0.03\% | 75.64±0.21\% | — | — |
| GraphSAGE | 92.24±0.07\% | 78.50±0.14\% | — | — |
| NeighborSampling | 91.70±0.09\% | 78.70±0.36\% | — | — |
| ClusterGCN | 92.12±0.09\% | 78.97±0.33\% | — | — |
| GraphSaint | — | 80.27±0.26\% | — | — |
| SIGN | 92.86±0.02\% | 80.52±0.13\% | 69.32±0.06\% | 65.68±0.06\% |
| SAGN | 93.09±0.04\% | 81.20±0.07\% | 70.34±0.99\% | 66.75±0.84\% |
| SAGN+1-SE | 92.54±0.04\% | 82.23±0.09\% | 70.79±0.12\% | 67.21±0.12\% |
| SAGN+2-SE | 92.33±0.03\% | 82.50±0.13\% | 70.89±0.12\% | 67.30±0.15\% |
| UniMP | 93.08±0.17\% | 82.56±0.31\% | 71.72±0.05\% | 67.36±0.10\% |
| MLP+C\&S | 91.47±0.09\% | 84.18±0.07\% | — | — |
| SAGN+0-SLE | 93.27±0.04\% | 83.29±0.18\% | 71.06±0.08\%  |  67.55±0.15\% |
| SAGN+1-SLE | 93.06±0.07\% | 84.18±0.14\% | 71.23±0.10\%  |  67.77±0.15\% |
| SAGN+2-SLE |  92.87±0.03\% | **84.28±0.14\%** | 71.31±0.10\%  |  **68.00±0.15\%** |
