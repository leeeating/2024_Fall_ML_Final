from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressorTrainer():
    def __init__(self, model, optimizer, device, args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.epoch = 1000
        self.criterion = nn.MSELoss()

    def train(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        for epoch in tqdm(range(self.epoch), desc="Regressor Training"):
            self.model.train()
            self.optimizer.zero_grad()
            pred_value = self.model(x)
            # print(pred_value.shape, y.shape)
            loss = self.criterion(pred_value, y)
            loss.backward()
            self.optimizer.step()
        return self.model.cpu()