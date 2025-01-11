import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GAE(torch.nn.Module):
    def __init__(self, encoder, edge_decoder, feat_decoder):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.edge_decoder = edge_decoder
        self.feat_decoder = feat_decoder

    def forward(self, x, edge_index):
        """
        used for evaluation
        """
        Z = self.encoder(x, edge_index)
        return Z
    
    def loss(self, x, edge_index):
        Z = self.encoder(x, edge_index)
        edge_score = self.edge_decoder(Z, edge_index)
        recon_feat = self.feat_decoder(Z)

        edge_label = torch.ones_like(edge_score).to(edge_score.device)
        edge_loss = F.binary_cross_entropy(edge_score, edge_label)
        feat_loss = F.mse_loss(recon_feat, x)
        return edge_loss, feat_loss
    
    @torch.no_grad()
    def get_embed(self, x, edge_index):
        self.eval()
        return self.encoder(x, edge_index)
    
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
        self.dropout = nn.Dropout()
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
    
class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index):
        # 使用隱藏表示z來計算邊的相似度（內積）
        z_i = z[edge_index[0]]
        z_j = z[edge_index[1]]
        edge_score = F.sigmoid((z_i * z_j).sum(dim=-1))
        return edge_score  # 內積
    
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout()
        self.act = nn.ReLU()

    def forward(self, embed):
        h = self.dropout(self.act(embed))
        score = self.lin(h)
        return score
    
class Regressor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_channels, int(hidden_channels/2)),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(int(hidden_channels/2), int(hidden_channels/4)),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(int(hidden_channels/4), out_channels)
        )
    def forward(self, x):
        return self.model(x).flatten()
    
    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self(x).flatten()
