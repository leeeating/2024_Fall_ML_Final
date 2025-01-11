import argparse
import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from LAL_base.model import *
from LAL_base import RepTrainer, RegressorTrainer
from LAL_base import LALDataset, ActiveLearner
from DataLoader import OriginalDataset
from Logger import Logger

parser = argparse.ArgumentParser()
# GAE parameters
parser.add_argument('--epochs', type=int, default=700)
parser.add_argument('--hidden', type=int, default=256)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-4)

parser.add_argument('--dataset', type=str, default='cora', help='dataset to be used')
parser.add_argument('--tag', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--budget', type=int, default=20)
parser.add_argument('--runs', type=int)
args = parser.parse_args()
device = torch.device('cuda')


dataset = OriginalDataset(tag=args.tag, device='cuda')
dataset.al_split(n_test=1000, n_val=500)
nfeat = dataset.x.shape[1]
# logger = Logger(args, data.name, model='gmlp', method='lal_ind', runs=args.runs)
# log_path = logger.path

label_size = range(2, 19)
repeats = range(50)
n_points_per_experiment = 10
BUDGET = args.budget * dataset.num_classes


encoder = GCNEncoder(nfeat, args.hidden).to(device)
edge_decoder = InnerProductDecoder().to(device)
feat_decoder = MLP(args.hidden, nfeat).to(device)
gae = GAE(encoder, edge_decoder, feat_decoder).to(device)
optimizer = Adam(gae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
EmbedTrainer = RepTrainer(gae, optimizer, device, args)
gae_model = EmbedTrainer.train(dataset)
embed = gae_model.get_embed(dataset.x, dataset.edge_index)
print(f"Embedding shape: {embed.shape}")

all_lal_feat = []
all_lal_labels = []
laldataset = LALDataset(embed, dataset, device, args)
with tqdm(total=len(label_size)*len(repeats), desc='laldataset') as pbar:
    for n_label in label_size:
        labeled_idx = dataset.init_split(run=0, init_num=n_label)
        cand_idx = dataset.cand_idx

        for num in repeats:
            # Expect different initial model weights
            best_model = laldataset.train(labeled_idx)
            
            random_selection_list = random.sample(cand_idx.tolist(), n_points_per_experiment)
            random_selection = torch.tensor(random_selection_list)
            batch_lal_feat, batch_lal_labels = laldataset.generate_lal_sample(best_model, labeled_idx, random_selection)        
            # print(batch_lal_feat.shape, batch_lal_labels.shape)
            all_lal_feat.append(batch_lal_feat)
            all_lal_labels.append(batch_lal_labels)
            pbar.update(1)

# (len(label_size)*len(repeats)*n_points_per_experiment, 2*num_classes + 1)
# maybe 9000 data points
all_lal_feat = torch.cat(all_lal_feat, dim=0)
all_lal_labels = torch.cat(all_lal_labels, dim=0)
print("LAL Dataset shape:")
print(all_lal_feat.shape, all_lal_labels.shape)
print()
torch.save(torch.cat([all_lal_feat, all_lal_labels.unsqueeze(1)], dim=1), f'./LALData/{dataset.name}_lal_dataset.pt')

regressor = Regressor(all_lal_feat.size(1), 64, 1).to(device)
optimizer = Adam(regressor.parameters(), lr=0.001, weight_decay=5e-4)
reg_trainer = RegressorTrainer(regressor, optimizer, device, args)
Selector = reg_trainer.train(all_lal_feat, all_lal_labels)

al_learner = ActiveLearner(embed, dataset, Selector, device, args)
test_acc_record = []
with tqdm(total=BUDGET, desc='Active Learning') as pbar:
    while len(al_learner.labeled_idx) < BUDGET:
        best_model = al_learner.train()
        test_acc, test_loss = al_learner.evaluation(best_model, embed, dataset.y, dataset.test_idx)
        test_acc_record.append(test_acc)
        al_learner.selectNext(best_model)
        pbar.update(1)
print(test_acc_record)
print(len(al_learner.labeled_idx))