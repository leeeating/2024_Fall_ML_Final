from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from .model import MLP
from DataLoader import OriginalDataset

class ActiveLearner:
    def __init__(self, embed, dataset, Selector, device, args):
        self.embeds = embed
        self.labels = dataset.y
        self.val_idx = dataset.val_idx
        self.test_idx = dataset.test_idx
        self.num_classes = dataset.num_classes
        
        self.Selector = Selector
        self.device = device
        self.args = args
        self.epoch = 300

        self.labeled_idx = dataset.init_split(run=99, init_num=2)
        self.cand_idx = dataset.cand_idx

    def selectNext(self, model):
        unknown_score = model(self.embeds[self.cand_idx])
        unknown_prob = F.softmax(unknown_score, dim=1).cpu().detach()
        lal_feat = self._getFeaturevector4LAL(unknown_prob, self.labeled_idx)
        lal_pred = self.Selector.predict(lal_feat) #(len(cand_idx),)
        select_key = torch.argmax(lal_pred)
        select_idx = self.cand_idx[select_key]

        self.labeled_idx = torch.cat([self.labeled_idx, select_idx.view(-1)])
        self.cand_idx = torch.cat([self.cand_idx[:select_key], self.cand_idx[select_key+1:]])

    def train(self):
        model = MLP(self.embeds.size(1), self.num_classes).to(self.device)
        optimizer  = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        best_model = None
        for epoch in range(self.epoch):
            model.train()
            optimizer.zero_grad()

            pred_score = model(self.embeds)
            loss = criterion(pred_score[self.labeled_idx], self.labels[self.labeled_idx])
            
            loss.backward()
            optimizer.step()
            eval_acc, eval_loss = self.evaluation(model, self.embeds, self.labels, self.val_idx)
            if eval_acc > best_val_acc:
                best_val_acc = eval_acc
                best_model = model
        return best_model
    
    @torch.no_grad()            
    def evaluation(self, model , embeds, labels, idx):
        model.eval()
        pred_score = model(embeds)
        pred_prob = F.softmax(pred_score, dim=1)
        pred_label = pred_prob.argmax(dim=1)
        acc = (pred_label[idx].eq(labels[idx])).sum().item() / len(idx)
        loss = F.cross_entropy(pred_score[idx], labels[idx])
        return acc, loss
    
    def _getFeaturevector4LAL(self, unknown_prob, labeled_idx):
        num_select = len(unknown_prob)
        num_labeled = len(labeled_idx)
        class_dis = torch.unique(self.labels[labeled_idx].cpu(), return_counts=True)[1] / num_labeled # (num_classes)
        f_1 = num_labeled * torch.ones(num_select, 1) # (num_select, 1)
        f_2 = class_dis * torch.ones(num_select, 1) # (num_select, num_classes)

        lal_feat = torch.cat([unknown_prob, f_1, f_2], dim=1) # (num_select, 2*num_classes + 1)
        return lal_feat