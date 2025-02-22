# Co-attention Graph Pooling for Efficient Pairwise Graph Interaction Learning
This repository provides an implementation of Co-Attention Graph Pooling (CAGPool) model in the paper **Co-attention Graph Pooling for Efficient Pairwise Graph Interaction Learning** (IEEE Access) by authors: Junhyun Lee*, Bumsoo Kim*, Minji Jeon, and Jaewoo Kang (*Equal contribution). 

<p align="center"><img width="100%" src="./imgs/CAGPool.png"></p>

## Abstract
Graph Neural Networks (GNNs) have proven to be effective in processing and learning from graph-structured data.
However, many real-world tasks require the analysis of pairs of graph-structured data, such as scene graph matching, code search, and drug-drug interaction prediction. 
Previous approaches for considering the interaction between these pairs of graphs have focused on the node level, leading to high computational costs and suboptimal performance. 
To address this issue, we propose a novel, efficient graph-level approach for extracting interaction representation using co-attention in graph pooling. 
Our method, Co-Attention Graph Pooling (CAGPool), shows competitive performance compared to existing methods on both classification and regression tasks on real-world datasets, while maintaining low computational complexity.

## Requirements
- Python 3.7 or higher
- PyTorch 1.7.0 or higher
- PyTorch Geometric
- numpy
- scipy

```bash
# Create rdkit environment
$ conda create -c rdkit -n rdkit-env python=3.6 rdkit
$ conda activate rdkit-env

# PyTorch 1.3.1
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# PyTorch Geometric
$ pip install --verbose --no-cache-dir torch-scatter
$ pip install --verbose --no-cache-dir torch-sparse
$ pip install --verbose --no-cache-dir torch-cluster
$ pip install --verbose --no-cache-dir torch-spline-conv (optional)
$ pip install torch-geometric

# PyYaml
$ pip install pyyaml
```

## Dataset
This implementation uses datasets stored in pickle format. The dataset should contain two kinds of information:
- The feature matrix of the nodes
- The adjacency matrices for each edge type

<p align="center"><img width="100%" src="./imgs/polypharmacy_graph.png"></p>

Download the dataset from the official [website](http://snap.stanford.edu/decagon/).
Dataset processing is handled by the `DecagonDataset_binary` and `DecagonDataset_multi` classes based on the selected label type (binary or multi).


## Setup

You need to create a configuration file (cfg) with all the necessary parameters and paths. The configuration file includes parameters for:

- Dataset paths
- Training parameters (e.g., learning rate, weight decay)
- Model parameters
- Whether you will use binary or multi label type
- Loss function choice



## Usage

To run the training phase:

```bash
python main.py --cfg path/to/config --phase train
```


To run the testing phase:

```bash
python main.py --cfg path/to/config --phase test
```


## Output

After running the script, the trained model will be saved to the directory specified in the configuration file. During the testing phase, the Area Under the Receiver Operating Characteristic (AUROC), Area Under Precision Recall Curve (AUPRC), and Average Precision at 50 (AP50) will be printed to the console.


## Citation
TBA
