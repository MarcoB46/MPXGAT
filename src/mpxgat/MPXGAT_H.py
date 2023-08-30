import torch
import torch.nn as nn
import torch.nn.functional as F
from mpxgat.multiplex_graph import MultiplexGraph
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class MPXGAT_H(nn.Module):
    def __init__(self, dataset: MultiplexGraph, in_channels=-1, hidden_channels=256, out_channels=256, heads=3):
        super(MPXGAT_H, self).__init__()
        self.num_layers = len(dataset.get_horizontal_layers())

        # List to store the GAT layers for each layer of the multiplex graph
        self.horizontal_layers_1 = nn.ModuleList([])
        self.horizontal_layers_2 = nn.ModuleList([])
        self.horizontal_layers_3 = nn.ModuleList([])

        # Add GAT layers for each layer of the multiplex graph
        for _ in range(self.num_layers):
            self.horizontal_layers_1.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.6, bias=True))
            self.horizontal_layers_2.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.6, bias=True))
            self.horizontal_layers_3.append(GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=0.6, bias=True))
       


    def forward(self, horizontal_graphs: list): 
        # Perform GAT convolution for each layer of the multiplex graph
        horizontal_layer_embeddings = []
        for i in range(self.num_layers):
            layer_graph: Data = horizontal_graphs[i]
            x = F.elu(self.horizontal_layers_1[i](layer_graph.x, layer_graph.edge_index))
            x = F.elu(self.horizontal_layers_2[i](x, layer_graph.edge_index))
            x = F.elu(self.horizontal_layers_3[i](x, layer_graph.edge_index))

            # append the embeddings of the current layer to the list of embeddings
            horizontal_layer_embeddings.append(x)

        # return the embeddings of each layer of the multiplex graph (this will be a list of tensors of size [num_nodes, out_channels])
        return horizontal_layer_embeddings

    # from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        
    # from https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py
    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()