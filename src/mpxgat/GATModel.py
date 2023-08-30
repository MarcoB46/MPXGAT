import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GAT(nn.Module):
    def __init__(self, in_channels=-1, hidden_channels=256, out_channels=32, heads=3):
        super(GAT, self).__init__()

        self.inter_layer_gat_1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6, bias=True, concat=True)
        self.inter_layer_gat_2 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6, bias=True, concat=True)
        self.inter_layer_gat_3 = GATConv(in_channels, out_channels, heads=heads, dropout=0.6, bias=True, concat=False)


    def forward(self, horizontal_layer_embeddings: list, vertical_graph: Data):
        x = F.elu(self.inter_layer_gat_1(vertical_graph.x, vertical_graph.edge_index))
        x = F.elu(self.inter_layer_gat_2(x, vertical_graph.edge_index))
        x = F.elu(self.inter_layer_gat_3(x, vertical_graph.edge_index))

        return x
    
    # from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    # from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()