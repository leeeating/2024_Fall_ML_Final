import argparse
from tqdm import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from DataLoader import OriginalDataset
from modules import GCNClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--dataset' ,type=str)
parser.add_argument('--model', type=str, default='mlp')
parser.add_argument('--init_node', type=int, default=10)

parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--budget', type=int, default=20)
parser.add_argument('--epochs', type=int, default=700)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

@torch.no_grad()
def evaluation(score, idx):
    pred = torch.argmax(score[idx], dim=1)
    acc = torch.sum(pred == data.y[idx]).item() / len(idx)
    return acc

data = OriginalDataset(tag=args.dataset, device=device)
data.al_split()
nfeat = data.x.shape[1]

BUDGET = args.budget * data.num_classes

train_idx = data.train_idx
randperm = torch.randperm(len(train_idx))
label_idx = train_idx[randperm[:BUDGET]]
print(f"Label idx: {label_idx}, len: {len(label_idx)}")

model = GCNClassifier(nfeat, args.hidden_dim, data.num_classes).to(device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_func = nn.CrossEntropyLoss()

best_val_acc = 0
for args.epochs in tqdm(range(args.epochs)):
    model.train()
    optimizer.zero_grad()
    
    embed, score = model(data.x, data.edge_index)
    loss = loss_func(score[label_idx], data.y[label_idx])
    
    loss.backward()
    optimizer.step()

    # val
    model.eval()
    _, score = model(data.x, data.edge_index)
    val_acc = evaluation(score, data.val_idx)
    
    if (val_acc > best_val_acc):
        best_val_acc = val_acc
        best_model = model

best_model.eval()
_, score = best_model(data.x, data.edge_index)
test_acc = evaluation(score, data.test_idx)
print(f"Test Accuracy: {test_acc}")