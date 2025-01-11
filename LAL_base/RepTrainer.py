import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from .model import GAE

class RepTrainer():
    def __init__(self, model:GAE, optimizer, device, args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.epoch = args.epochs

    def train(self, dataset):
        for epoch in tqdm(range(self.epoch), desc='Structure Learning'):
            self.model.train()
            self.optimizer.zero_grad()

            edge_loss, feat_loss = self.model.loss(dataset.x, dataset.edge_index)
            loss = edge_loss + 0.01*feat_loss
            loss.backward() #type: ignore
            self.optimizer.step()
        return self.model
