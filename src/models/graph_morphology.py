import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """Single graph convolution layer."""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """x: [N, F], adj: [N, N]"""
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MorphologyGNN(nn.Module):
    """GNN for morphology classification."""
    def __init__(self, nfeat=64, nhid=128, nclass=3, dropout=0.5):
        super(MorphologyGNN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """x: [N, F], adj: [N, N]"""
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Global mean pooling
        x = torch.mean(x, dim=0, keepdim=True)
        
        x = self.fc(x)
        return x
