import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATConv, global_mean_pool, ASAPooling, GCN2Conv


class MSSN(nn.Module):
    def __init__(self, esm_dim=1024, hidden_dim=512, num_classes=320):
        super().__init__()
        self.proj = nn.Linear(esm_dim, hidden_dim)
        d2 = hidden_dim // 2
        self.sc = nn.Sequential(nn.Conv1d(hidden_dim, d2, 1, bias=False),
                                nn.BatchNorm1d(d2))
        self.conv1 = nn.Sequential(nn.Conv1d(hidden_dim, d2, 1, bias=False),
                                   nn.BatchNorm1d(d2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(hidden_dim, d2, 6, padding=3, bias=False),
                                   nn.BatchNorm1d(d2), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(hidden_dim, d2, 12, padding=6, bias=False),
                                   nn.BatchNorm1d(d2), nn.ReLU())
        self.classifier = nn.Linear(d2*3, num_classes)

    def forward(self, data):
        x = F.relu(self.proj(data.x.float()))       # [N,H]
        xB, _ = to_dense_batch(x, data.batch)       # [B,L,H]
        xT = xB.permute(0,2,1)                      # [B,H,L]
        idt = self.sc(xT)                           # [B,d2,L]
        L = idt.size(2)
        y1 = self.conv1(xT)[:,:,:L] + idt
        y3 = self.conv3(xT)[:,:,:L] + idt
        y5 = self.conv5(xT)[:,:,:L] + idt
        p1 = F.adaptive_max_pool1d(y1,1).squeeze(-1)
        p3 = F.adaptive_max_pool1d(y3,1).squeeze(-1)
        p5 = F.adaptive_max_pool1d(y5,1).squeeze(-1)
        feat = torch.cat([p1,p3,p5], dim=1)         # [B,3*d2]
        logit = self.classifier(feat)               # [B,C]
        return feat, logit


class MTGN(nn.Module):
    def __init__(self, input_dim=1024, in_dim=512, num_classes=320):
        super().__init__()
        self.lin  = nn.Linear(input_dim, in_dim)
        self.gcn1 = GCN2Conv(in_dim, alpha=0.3, theta=1.0, layer=1)
        self.gcn2 = GCN2Conv(in_dim, alpha=0.3, theta=1.0, layer=3)
        self.gat  = GATConv(in_dim, in_dim//4, heads=4, concat=True)
        self.pool = ASAPooling(in_dim, ratio=0.8)
        self.attn = nn.MultiheadAttention(in_dim, num_heads=2, batch_first=True)
        self.cls  = nn.Linear(in_dim, num_classes)

    def forward(self, data):
        x = F.relu(self.lin(data.x.float()))
        edge_index = data.edge_index
        edge_weight = torch.ones(edge_index.size(1), device=x.device)
        h1 = self.gcn1(x, x, edge_index, edge_weight)
        h1 = F.relu(h1 + x)
        h2 = self.gcn2(h1, h1, edge_index, edge_weight)
        h2 = F.relu(h2 + h1)
        h3 = F.relu(self.gat(x, edge_index))
        xp, _, _, batch_p, _ = self.pool(x, edge_index, batch=data.batch)
        h4 = F.relu(xp)
        p1 = global_mean_pool(h2, data.batch)
        p2 = global_mean_pool(h3, data.batch)
        p3 = global_mean_pool(h4, batch_p)
        feats = torch.stack([p1,p2,p3], dim=1)
        a,_   = self.attn(feats, feats, feats)
        fused= a.mean(dim=1)
        logit= self.cls(fused)
        return fused, logit

class SeqGraphAF(nn.Module):
    def __init__(self, esm_dim=1024, aa_vocab=21,
                 hidden_dim=512, num_classes=320):
        super().__init__()
        self.e1 = MSSN(esm_dim, hidden_dim, num_classes)

        self.e2 = MTGN(esm_dim, hidden_dim, num_classes)
    
    def forward(self, data):
        f1, l1 = self.e1(data)  # [B,fe1], [B,C]

        f2, l2 = self.e2(data)

        logits_stack = torch.stack([l1, l2], dim=1)  # [B,3,C]
        combined = logits_stack.mean(dim=1)               # [B,C]
        # return combined, l1, l2, l3
        return combined, l1, l2
