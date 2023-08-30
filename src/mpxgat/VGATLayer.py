from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
import matplotlib.pyplot as plt


class VGATLayer(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        include_horizontal: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.betaPlotInfo = []
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.include_horizontal = include_horizontal

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        # NOTE: see https://pytorch.org/docs/stable/generated/torch.nn.Linear.html for the Linear class
        if isinstance(in_channels, int): 
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=True, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, True,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, True,
                                  weight_initializer='glorot')
                                    
        self.lin_horizontal = Linear(in_channels, heads * out_channels, True,
                                    weight_initializer='glorot')
        self.att_horizontal = Parameter(torch.Tensor(1, heads, out_channels))
        
        self.horizontal_regulator = Parameter(torch.Tensor(1, heads, 1)) # this is my beta parameter
        
       

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim != None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge != None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)
        glorot(self.horizontal_regulator)
        self.lin_horizontal.reset_parameters()
        glorot(self.att_horizontal)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, 
                horizontal_layers_embedding: Tensor,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels        
        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst != None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
        if self.include_horizontal:
            x_horizontal = self.lin_horizontal(horizontal_layers_embedding).view(-1, H, C)
            alpha_horizontal = (x_horizontal * self.att_horizontal).sum(dim=-1)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        # ! alpha_src and alpha_dst are the attention coefficients for the vertical embedding (i.e. the embedding of the node itself)
        # ! they will be summed togheter in the edge_update function, where the rest of the formula is computed.
        alpha_src = (x_src * self.att_src).sum(dim=-1)    
        alpha_dst = None if x_dst == None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        
        x = (x_src, x_dst)
        
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst != None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size != None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim == None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        # this is not properly a "self attention mechanism", it is just a weigh used to sum the vertcal embedding and the horizontal embedding
        alpha_horizontal = None if alpha_horizontal == None else F.leaky_relu(alpha_horizontal, self.negative_slope)
        # ! alpha_horizontal is not manipulated by the edge_updater function
        alpha, alpha_horizontal = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, alpha_H=alpha_horizontal if self.include_horizontal else None)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size, alpha_horizontal=alpha_horizontal, x_H = x_horizontal, beta = self.horizontal_regulator)
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias != None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int], alpha_H: Tensor) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i == None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr != None and self.lin_edge != None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge
        # ! this is the rest of the formula, leacky relu, softmax and dropout.
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        
        return alpha, alpha_H

    def message(self, x_j: Tensor, alpha: Tensor, alpha_horizontal_j: Tensor, x_H_j: Tensor, beta: Tensor) -> Tensor:
        return ((alpha.unsqueeze(-1) * x_j) * (1 - F.relu(beta))) + (alpha_horizontal_j.unsqueeze(-1) * x_H_j * F.relu(beta))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
        
    def printBetaPlotInfo(self, fileDest:str, layer: 0):
        # if the model is not in training mode, return
        if not self.training:
            return
        # plot the beta values
        epoch_list = range(len(self.betaPlotInfo))
        # create a plot with the loss and the AUC values
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        # each line refers to a different epoch and a different head
        for head in range(self.heads):
            ax.plot(epoch_list, [x[head].detach().cpu().numpy() for x in self.betaPlotInfo], label=f'head {head}')
        # print the layer info as legend
        ax.set_title(f'Layer {layer}')
        ax.legend()
        # fig.show()
        fig.savefig(fileDest+str(layer)+'.png')
