import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution Layer (GCN)
    Allows node features to be updated based on their neighbors' features.
    """
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
        """
        x: Node features [N, in_features]
        adj: Adjacency matrix [N, N]
        """
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MorphologyGNN(nn.Module):
    """
    Graph Neural Network for Morphology Classification.
    Represents the fungal/bacterial grain structures (filaments, branching)
    as graphs where nodes are morphological keypoints and edges denote connectivity.
    """
    def __init__(self, nfeat=64, nhid=128, nclass=3, dropout=0.5):
        """
        Args:
            nfeat: Input node feature dimension
            nhid: Hidden dimension
            nclass: Output morphology classes (e.g., fungal, bacterial, unknown)
        """
        super(MorphologyGNN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        Forward pass for a single graph instance.
        x: Node feature matrix [N_nodes, nfeat]
        adj: Adjacency matrix [N_nodes, N_nodes]
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Readout (Global mean pooling over all nodes)
        x = torch.mean(x, dim=0, keepdim=True)
        
        # Classify graph
        x = self.fc(x)
        return x
