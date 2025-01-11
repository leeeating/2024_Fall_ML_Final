# type: ignore

import numpy as np
import torch
import argparse
import scipy.io as sio
import random
import os

from sklearn import preprocessing
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import contains_isolated_nodes

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')  #'BlogCatalog'  'Flickr' 'cora'  'citeseer'  'pubmed' 
parser.add_argument('--seed', type=int, default=1)  #random seed
parser.add_argument('--p', type=int, default=15)  #num of fully connected nodes
parser.add_argument('--n', type=int)  
parser.add_argument('--k', type=int, default=50)  #num of clusters
args = parser.parse_args()

dataset_str = args.dataset  #'BlogCatalog', 'Flickr', 'cora', 'citeseer', 'pubmed'
seed = 0
cluster_size = args.p  #num of fully connected nodes 
attr_affect_size = args.k

if args.n is None:
    if dataset_str == 'cora' or dataset_str == 'citeseer':
        n_cluster = 5
    elif dataset_str == 'blogcatalog':
        n_cluster = 10
    elif dataset_str == 'flickr':
        n_cluster = 15
    elif dataset_str == 'pubmed':
        n_cluster = 20
else:
	n_cluster = args.n

print('Random seed: {:d}. \n'.format(seed))
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

print('Loading Data')
if args.dataset in ['cora', 'pubmed', 'citeseer']:
    dataset = Planetoid(root='./Data', name=args.dataset)
    data = dataset[0]
    feat = data.x.numpy()
    class_label = data.y.numpy()
    edge_index = data.edge_index
    print(f"Dataset has isolated nodes : {contains_isolated_nodes(edge_index)}")
    n_nodes = feat.shape[0]
    adj_dense = torch.sparse_coo_tensor(indices=edge_index, 
                                        values=torch.ones(edge_index.shape[1]),
                                        size=(n_nodes, n_nodes)).to_dense().numpy()
else:
    # 'BlogCatalog', 'Flickr'
    data = sio.loadmat(f'./Data/{args.dataset}.mat')
    feat = np.array(data['Attributes'].todense())
    # l2-norm
    feat = preprocessing.normalize(feat, axis=0)
    adj_dense = np.array(data['Network'].todense())
    class_label = data['Label']
    class_label = (class_label-1).flatten() # 1-based to 0-based

ori_num_edge = np.sum(adj_dense)
num_node = adj_dense.shape[0]
print('Done. \n')

# Random pick anomaly nodes
all_idx = list(range(num_node))
random.shuffle(all_idx)
anomaly_idx = all_idx[:cluster_size*n_cluster*2]
structure_anomaly_idx = anomaly_idx[:cluster_size*n_cluster]
attribute_anomaly_idx = anomaly_idx[cluster_size*n_cluster:]

ano_label = torch.zeros((num_node), dtype=torch.long)
ano_label[anomaly_idx] = 1
str_anomaly_label = torch.zeros((num_node), dtype=torch.long)
str_anomaly_label[structure_anomaly_idx] = 1
attr_anomaly_label = torch.zeros((num_node), dtype=torch.long)
attr_anomaly_label[attribute_anomaly_idx] = 1


# Disturb structure
print('Constructing structured anomaly nodes...')
for n_ in range(n_cluster):
    current_nodes = structure_anomaly_idx[n_*cluster_size:(n_+1)*cluster_size]
    for i in current_nodes:
        for j in current_nodes:
            adj_dense[i, j] = 1.

    adj_dense[current_nodes,current_nodes] = 0.

num_add_edge = int(np.sum(adj_dense) - ori_num_edge)
print(f'Done. {len(structure_anomaly_idx)} structured nodes are constructed. ({num_add_edge} edges are added) \n')

# Disturb attribute
print('Constructing attributed anomaly nodes...')
for i_ in attribute_anomaly_idx:
    picked_list = random.sample(all_idx, attr_affect_size)
    max_dist = 0
    for j_ in picked_list:
        cur_dist = np.sqrt(np.sum((feat[i_] - feat[j_])**2, axis=-1))
        if cur_dist > max_dist:
            max_dist = cur_dist
            max_idx = j_

    feat[i_] = feat[max_idx]
print('Done. {:d} attributed nodes are constructed. \n'.format(len(attribute_anomaly_idx)))

# Pack & save them into .mat
print('Saving mat file...')
row, col = adj_dense.nonzero()
edge_index = torch.from_numpy(np.vstack([row, col]))
print(edge_index.shape)
data = Data(x=torch.from_numpy(feat), ano_label=ano_label, class_label=torch.from_numpy(class_label), edge_index=edge_index, 
            str_anomaly_label=str_anomaly_label, attr_anomaly_label=attr_anomaly_label,
            num_add_edge=num_add_edge, p=cluster_size, q=n_cluster)

# torch.save(data, f"./Data/injected/{args.dataset}.pt")
print('Done. The file is save as: dataset/{}.mat \n'.format(dataset_str))

