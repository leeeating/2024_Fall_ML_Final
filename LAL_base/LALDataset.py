from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from .model import MLP
from DataLoader import OriginalDataset

class LALDataset:
    def __init__(self, embed, dataset:OriginalDataset, device, args):
        self.embeds = embed
        self.labels = dataset.y
        self.val_idx = dataset.val_idx
        self.test_idx = dataset.test_idx
        self.num_classes = dataset.num_classes
        
        self.device = device
        self.args = args
        self.epoch = 200
        
    def generate_lal_sample(self, model, labeled_idx, rand_select_idx):
        """
        lal_feat: (n_points_per_experiment, 2*num_classes + 1)
        lal_labels: (n_points_per_experiment)
        """
        prev_test_acc, prev_test_loss = self.evaluation(model, self.embeds, self.labels, self.test_idx)
        
        unknown_score = model(self.embeds[rand_select_idx])
        unknown_prob = F.softmax(unknown_score, dim=1).cpu().detach() # (n_points_per_experiment, num_classes) 
        lal_feat = self._getFeaturevector4LAL(unknown_prob, labeled_idx)
        
        lal_labels = torch.zeros(len(rand_select_idx))        
        for key, idx in enumerate(rand_select_idx):
            tmp_label_idx = torch.cat([labeled_idx, idx.view(-1)])
            best_model = self.train(tmp_label_idx)
            test_acc, test_loss = self.evaluation(best_model, self.embeds, self.labels, self.test_idx)
            lal_labels[key] = test_acc - prev_test_acc
        return lal_feat, lal_labels
    
    def train(self, labeled_idx):
        classifier = MLP(self.embeds.size(1), self.num_classes).to(self.device)
        optimizer  = Adam(classifier.parameters(), lr=0.001, weight_decay=5e-3)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        best_model = None
        for epoch in range(self.epoch):
            classifier.train()
            optimizer.zero_grad()

            pred_score = classifier(self.embeds)
            loss = criterion(pred_score[labeled_idx], self.labels[labeled_idx])
            
            loss.backward()
            optimizer.step()
            eval_acc, eval_loss = self.evaluation(classifier, self.embeds, self.labels, self.val_idx)
            if eval_acc > best_val_acc:
                best_val_acc = eval_acc
                best_model = classifier
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