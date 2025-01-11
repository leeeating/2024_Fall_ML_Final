# type:ignore

import numpy as np
import torch
import scipy.io as sio
import random
import torch_sparse

from torch_geometric.datasets import HeterophilousGraphDataset, Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import contains_self_loops, add_remaining_self_loops, is_undirected
from sklearn.model_selection import train_test_split


class OriginalDataset():

    def __init__(self, device, tag=None, name=None):
        self.device = device
        self.name = name
        if tag is not None:
            dataset = {1:'cora', 2:'citeseer', 3:'pubmed', 
                       4:'flickr', 5:'blogcatalog'}
            self.name = dataset.get(int(tag))
        
        print(f"Loading {self.name} Data")
        # geo Data()
        data = self._load_dataset()

        self.x = data['x'].to(device)
        self.y = data['y'].to(device)
        self.edge_index = data['edge_index'].to(device)
        self.data = Data(x=self.x, y=self.y , edge_index=self.edge_index)

        self.num_classes = len(torch.unique(self.y))
        self.num_nodes = self.x.shape[0]
        self.num_edges = self.edge_index.shape[1]

        set_seed(0)
        self.split_seed_list = random.sample(range(1, 1_000_001), 100)
        average_degree = self.num_edges / self.num_nodes
        self.neig_sample = self._get_neighor_sample(average_degree)
        

        print(f"Feature shape : {self.x.shape}")
        print(f"Number of edge : {self.num_edges}")
        print(f"avgerage degree : {average_degree:.2f}")
        print(f"Number of class : {self.num_classes}")

    def al_split(self, n_test=1000, n_val=500, seed=0):
        """
        Splitting the dataset into train/test/val
        seed : constant seed for reproducibility
        """
        set_seed(seed)
        all_idx = torch.arange(self.num_nodes)
        train_idx, test_val_idx = train_test_split(all_idx, test_size=n_test+n_val, stratify=self.y.cpu(), random_state=seed)
        test_idx, val_idx = train_test_split(test_val_idx, test_size=n_val, stratify=self.y[test_val_idx].cpu(), random_state=seed)

        self.train_idx = train_idx 
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.n_train = len(self.train_idx)
        self.n_test = len(self.test_idx)
        self.n_val = len(self.val_idx)
        print(f"Number of node in train/test/val : {self.n_train}/{self.n_test}/{self.n_val}")
        
        train_dis = torch.unique(self.y[self.train_idx], return_counts=True)[1] / self.n_train
        val_dis = torch.unique(self.y[self.val_idx], return_counts=True)[1] / self.n_val
        test_dis = torch.unique(self.y[self.test_idx], return_counts=True)[1] / self.n_test
        print(f"Val Class Dis., {[f"{x:.4f}" for x in train_dis.tolist()]}")
        print(f"Val Class Dis., {[f"{x:.4f}" for x in val_dis.tolist()]}")
        print(f"Test Class Dis., {[f"{x:.4f}" for x in test_dis.tolist()]}")

    def init_split(self, run, init_num):
        cur_seed = self.split_seed_list[run]
        print(f"Split init node in each class with seed {cur_seed}")
        random.seed(cur_seed)

        label_position = self._group_train_idx(self.y, self.train_idx)
            
        selection_positions_list = []
        for label, keys in label_position.items():
            if len(keys) > init_num:
                selection_positions_list.extend(random.sample(keys, k=init_num))
            else:
                selection_positions_list.extend(keys)

        # check duplicate
        assert len(torch.unique(self.train_idx[selection_positions_list])) == len(self.train_idx[selection_positions_list])

        labeled_idx = self.train_idx[selection_positions_list]
        tmp_mask = torch.isin(self.train_idx, labeled_idx, invert=True)
        self.cand_idx = self.train_idx[tmp_mask]
        return labeled_idx

    def _group_train_idx(self, label, idx):
        """
        Grouping the training index by label
        position means the index in the training data
        """
        label_position = {}
        for position, label in enumerate(label[idx]):
            if label.item() not in label_position:
                label_position[label.item()] = []
            label_position[label.item()].append(position)
        return label_position

    def _load_dataset(self):
        """
        Loading normalized data
        """
        data = {}
        if self.name in ['cora', 'citeseer', 'pubmed']:

            dataset = Planetoid(root='./Data', name=self.name)
            data = dataset[0]
            data['x'] = data.x
            data['y'] = data.y
            data['edge_index'] = data.edge_index
        else:
            data = sio.loadmat(f'./Data/{self.name}.mat')
            features = torch.tensor(data['Attributes'].toarray(), dtype=torch.float32)
            data['x'] = row_normalize(features)
            label = (data['Label'] - 1.0).flatten()
            data['y'] = torch.tensor(label, dtype=torch.int64)
            src, dst = data['Network'].nonzero()
            edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
            data['edge_index'] = edge_index

        return data
    
    def _get_neighor_sample(self, average_degree):
        if average_degree < 10:
            return [20, 10]
        elif average_degree < 50:
            return [15, 10]
        else:
            return [10, 5]

class HeterDataset():

    def __init__(self, device, mask=0, tag=None, name=None):
        super().__init__()

        self.name = name
        if tag is not None:
            dataset = {1:'questions', 2:'tolokers', 
                    3:'amazon-upu', 4:'amazon-usu', 5:'amazon-uvu', 
                    6:'yelpchi-rtr', 7:'yelpchi-rsr'}
            self.name = dataset.get(int(tag))
        
        print(f"Loading {self.name} Data")
        self.device = device
        # Data(x, edge_index, ano_label, train_mask, test_mask, val_maks)
        if self.name in ['questions', "tolokers"]:
            dataset = HeterophilousGraphDataset(root='/home/leeating/root/Mymodel/Data', name=self.name)
            data = dataset[0]
            self.data = self.mask_selection(data, mask=mask)
        else:
            dataset = FraudDataset(name=self.name)
            data = dataset.data

        maj, mino = torch.unique(data.ano_label, return_counts=True)[1]

        self.x = data.x.to(device)
        self.ano_label = data.ano_label.float().to(device)
        self.edge_index = data.edge_index.to(device)
        self.pos_ratio = float(mino/len(data.ano_label))
        self.n_nodes = self.x.shape[0]

        self.num_classes = 2
        self.model_out = 1

        self.train_idx = torch.nonzero(data.train_mask).flatten().to(device)
        self.test_idx = torch.nonzero(data.test_mask).flatten()
        self.val_idx = torch.nonzero(data.val_mask).flatten()

        self.train_mask = torch.zeros_like(self.ano_label, dtype=bool)
        self.train_mask[self.train_idx] = True
        self.n_train = len(self.train_idx)
        self.n_test = len(self.test_idx)
        self.n_val = len(self.val_idx)

        print(f"Imablance Rate:{float(maj/mino):.2f}, Minority Percnetage:{self.pos_ratio*100:.2f}")
        print(f'Numbers of node in each class : {maj}, {mino}')
        print(f"Number of node in train/test/val : {self.n_train}/{self.n_test}/{self.n_val}")
        print(f"Number of edge : {self.edge_index.shape[1]}")

    def mask_selection(self, data, mask):
        data.train_mask = data.train_mask[:, mask]
        data.test_mask = data.test_mask[:, mask]
        data.val_mask = data.val_mask[:, mask]
        return data

    def al_split(self):
        new_train_idx, new_test_idx, new_val_idx = self.ttv_split()
        self.train_idx = new_train_idx.to(self.device)
        self.test_idx = new_test_idx
        self.val_idx = new_val_idx

        self.train_mask = torch.zeros_like(self.ano_label, dtype=bool)
        self.train_mask[self.train_idx] = True
        self.n_train = len(self.train_idx)
        self.n_test = len(self.test_idx)
        self.n_val = len(self.val_idx)
        print("Splitting train/test/val")
        print(f"Number of nodes in train/test/val : {self.n_train}/{self.n_test}/{self.n_val}")

    def ttv_split(self):
        # test_idx = torch.nonzero(data.test_mask).flatten()
        # val_idx = torch.nonzero(data.val_mask).flatten()
        pos_idx = torch.nonzero(self.ano_label).flatten()

        new_test_idx = self.pos_neg_selection(self.test_idx, pos_idx, size=1000)
        new_val_idx = self.pos_neg_selection(self.val_idx, pos_idx, size=500)
        node_idx = torch.arange(len(self.ano_label))
        new_train_idx = torch.isin(node_idx, torch.hstack([new_test_idx, new_val_idx]), invert=True).nonzero().flatten()

        return new_train_idx, new_test_idx, new_val_idx

    def pos_neg_selection(self, idx, pos_idx, size=500):
        pos_size = int(size * self.pos_ratio)
        neg_size = size - pos_size

        pos_mask = torch.isin(idx, pos_idx)
        pos = pos_mask.nonzero().flatten()
        neg = (~pos_mask).nonzero().flatten()

        pos_key = torch.randperm(len(pos))[:pos_size]
        neg_key = torch.randperm(len(neg))[:neg_size]

        sele_idx = torch.hstack([idx[pos[pos_key]], idx[neg[neg_key]]])

        return sele_idx

    def init_split(self, run, k=10, pos_k=None, neg_k=None):
        if pos_k is None and neg_k is None:
            pos_k = neg_k = k

        print('Split init node in each class')
        train_idx = self.train_idx.cpu().numpy()
        labels = self.ano_label.cpu().numpy()

        train_mask = np.zeros_like(labels, dtype=bool)
        train_mask[train_idx] = True
        pos_mask = (labels == 1)

        pos_train = (pos_mask & train_mask).nonzero()[0]
        neg_train = (~pos_mask & train_mask).nonzero()[0]

        seed = random.sample(range(1, 10001), 10)
        random.seed(seed[run])
        sel_pos = random.sample(pos_train.tolist(), k=pos_k)
        sel_neg = random.sample(neg_train.tolist(), k=neg_k)

        return torch.tensor([sel_pos + sel_neg]).flatten()
    
    def get_log_path(self, model, method):

        import os 
        import time

        name = f"{model}-{method}"
        rec = time.strftime("%m%d")
        idx = 1
        while os.path.exists(f'./log/{name}/{self.name}/{rec}-{idx}'):
            idx += 1
        os.makedirs(f'log/{name}/{self.name}/{rec}-{idx}')
        print('Experiment data saving in :', f'./log/{name}/{self.name}/{rec}-{idx}')

        wb_group = time.strftime("%ano_label-%m-%d") + f'-{idx}'
        return f'log/{name}/{self.name}/{rec}-{idx}', idx, wb_group

class InjectedDataset():

    def __init__(self, device, injected=True, tag=None, name=None):

        self.name = name
        if tag is not None:
            dataset = {1:'cora', 2:'citeseer', 
                    3:'pubmed', 4:'flickr', 5:'blogcatalog'}
            self.name = dataset.get(int(tag))
        
        print(f"Loading {self.name} Data")
        self.device = device
        # geo Data()
        pt_data = torch.load(f"./Data/injected/{self.name}.pt")

        self.x = row_normalize(pt_data.x).to(device)
        self.ano_label = pt_data.ano_label
        self.class_label = pt_data.class_label.to(device) 
        # if self.name  in ['flickr', 'blogcatalog']:
        #     self.class_label = (data.class_label-1).to(device).squeeze()
        self.edge_index = pt_data.edge_index.to(device)
        
        maj, mino = torch.unique(self.ano_label, return_counts=True)[1]
        self.ano_ratio = float(mino/len(self.ano_label))

        self.n_nodes = self.x.shape[0]
        self.n_class = len(torch.unique(self.class_label))
        self.data = Data(x=self.x, edge_index=self.edge_index)

        label_dis = torch.unique(self.class_label, return_counts=True)[1] / self.n_nodes

        print(f"Number of anomaly nodes : {int(pt_data.p * pt_data.q * 2)}")
        print(f"Imablance Rate:{float(maj/mino):.2f}, Minority Percnetage:{self.ano_ratio*100:.2f}")
        print(f'Numbers of node in Norm/Anom : {maj}, {mino} ({len(pt_data.x)})')
        print(f"Number of edge after injected : {self.edge_index.shape[1]}(injected edge : {pt_data.num_add_edge})")
        print(f'Original Class Dis., {[round(x*100, 2) for x in label_dis.tolist()]}')
        print()

    def al_split(self, n_test=1000, n_val=500):
        ano_label = self.ano_label.cpu()
        ano_idx = torch.where(ano_label == 1)[0]
        nor_idx = torch.where(ano_label == 0)[0]
        torch.manual_seed(0)
        ano_idx = ano_idx[torch.randperm(len(ano_idx))]
        nor_idx = nor_idx[torch.randperm(len(nor_idx))]
        # print(ano_idx)

        n_test_ano, n_val_ano = int(n_test * self.ano_ratio), int(n_val * self.ano_ratio)
        n_test_nor, n_val_nor = n_test-n_test_ano, n_val-n_val_ano

        test_idx = torch.hstack([ano_idx[:n_test_ano], nor_idx[:n_test_nor]])
        val_idx = torch.hstack([ano_idx[n_test_ano:n_test_ano+n_val_ano], nor_idx[n_test_nor:n_test_nor+n_val_nor]])
        train_idx = torch.hstack([ano_idx[n_test_ano+n_val_ano:], nor_idx[n_test_nor+n_val_nor:]])

        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.n_train = len(self.train_idx)
        self.n_test = len(self.test_idx)
        self.n_val = len(self.val_idx)
        print(f"Number of node in train/test/val : {self.n_train}/{self.n_test}/{self.n_val}")
        
        train_dis = torch.unique(self.class_label[self.train_idx], return_counts=True)[1] / self.n_train
        val_dis = torch.unique(self.class_label[self.val_idx], return_counts=True)[1] / self.n_val
        test_dis = torch.unique(self.class_label[self.test_idx], return_counts=True)[1] / self.n_test
        print(f"Train Class Dis., {[round(x*100, 2) for x in train_dis.tolist()]}")
        print(f"Val Class Dis., {[round(x*100, 2) for x in val_dis.tolist()]}")
        print(f"Test Class Dis., {[round(x*100, 3) for x in test_dis.tolist()]}")

    def init_split(self, run, init_num):
        random.seed(0)
        seed = random.sample(range(1, 10001), 100)
        # print(seed)
        random.seed(seed[run])

        label_position = {}
        for key, label in enumerate(self.class_label[self.train_idx]):
            if label.item() not in label_position:
                label_position[label.item()] = []
            label_position[label.item()].append(key)
            
        random_positions_list = []
        for label, keys in label_position.items():
            if len(keys) > init_num:
                random_positions_list.extend(random.sample(keys, k=init_num))
            else:
                random_positions_list.extend(keys)

        # check duplicate
        assert len(torch.unique(self.train_idx[random_positions_list])) == len(self.train_idx[random_positions_list])
        return self.train_idx[random_positions_list]

    def get_A_square(self):
        edge_index = self.edge_index
        edge_weight = torch.ones(edge_index.shape[1]).to(self.device)
        A = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=(self.n_nodes, self.n_nodes))
        A_square = torch.sparse.mm(A, A)
        return A.to_dense(), A_square.to_dense()

class FraudDataset(object):

    def __init__(self, name):
        """
        name : [Amazon-upu, Amazon-usu, Amazon-uvu, YelpChi]
        """
        name, net = name.split('-')
        self.name = name
        self.net = net
        self.load_data()
        self.split()
        self.data = Data(x=self.features, edge_index=self.edge_index, ano_label=self.label, 
                        train_mask=self.train_mask, test_mask=self.test_mask, val_mask=self.val_mask)


    def load_data(self):
        path = f'./Data/{self.name}.mat'
        relation = f'net_{self.net}'
        self.features, self.edge_index, self.label = load_fraud_mat(path, relation)

    def split(self):

        l = self.label.shape[0]
        if self.name == 'yelpchi':
            index = list(range(l))
            idx_train_tmp, idx_test, y_train_tmp, _ = train_test_split(index, self.label, stratify = self.label, train_size=0.75, random_state=2, shuffle=True)
            idx_train, idx_val, _, _ = train_test_split(idx_train_tmp, y_train_tmp, stratify = y_train_tmp, train_size=0.66, random_state=2, shuffle=True)

        elif self.name == 'amazon':  # amazon
            # 0-3304 are unlabeled nodes
            index = list(range(3305, l))
            idx_train_tmp, idx_test, y_train_tmp, _ = train_test_split(index, self.label[3305:], stratify = self.label[3305:], train_size=0.75, random_state=2, shuffle=True)
            idx_train, idx_val, _, _ = train_test_split(idx_train_tmp, y_train_tmp, stratify = y_train_tmp, train_size=0.66, random_state=2, shuffle=True)


        self.train_mask = torch.zeros_like(self.label, dtype=bool)
        self.test_mask = torch.zeros_like(self.label, dtype=bool)
        self.val_mask = torch.zeros_like(self.label, dtype=bool)
        
        self.train_mask[idx_train] = True
        self.test_mask[idx_test] = True 
        self.val_mask[idx_val] = True               



def load_fraud_mat(path, relation):

    data = sio.loadmat(path)

    features = torch.tensor(data['features'].todense(), dtype=torch.float32)

    src, dst = data[relation].nonzero()
    edge_index = torch.tensor(np.vstack([src ,dst]), dtype=torch.long)

    label = torch.tensor(data['label'].flatten(), dtype=torch.long)

    return features, edge_index, label

def row_normalize(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = features.sum(1)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    features = r_mat_inv.mm(features)
    return features.to(torch.float32)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False