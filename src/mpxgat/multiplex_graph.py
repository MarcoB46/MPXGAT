import torch
import time    
from torch_geometric.data import Data
from torch_geometric.utils import degree
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import os


class MultiplexGraph:
    def __init__(self, horizontal_layers: list[Data]=None, vertical_graph: Data=None, node_identifier_map=None, graphName=None) -> None:
        self.vertical_graph = vertical_graph
        self.horizontal_layers = horizontal_layers
        self.node_identifier_map = node_identifier_map
        if graphName == None:
            self.graph_name = "random_graph_" + time.strftime("%y-%m-%d-%H:%M:%S")
        else:
            self.graph_name = graphName
    
    
    def get_horizontal_layers(self):
        return self.horizontal_layers
    
    def get_vertical_graph(self):
        return self.vertical_graph
    
    def save(self, path):
        print("Saving graph to " + path + self.graph_name)
        # create the folder on the given path if those don't exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self, path)
        
    def load(self, path):
        print("Loading graph from " + path)
        return torch.load(path)
    
    def set_transform(self, transform):
        self.transform = transform
        
        if self.transform != None:
            for i in range(len(self.horizontal_layers)):
                # convert the graph labels to Float Tensors, this could be necessary if the dataset is not already in Float Tensors
                if self.horizontal_layers[i].y != None:
                    self.horizontal_layers[i].y = self.horizontal_layers[i].y.float()
                if self.horizontal_layers[i].edge_attr != None:
                    self.horizontal_layers[i].edge_attr = self.horizontal_layers[i].edge_attr.float()
                if self.horizontal_layers[i].x != None:
                    self.horizontal_layers[i].x = self.horizontal_layers[i].x.float()
                    
                self.horizontal_layers[i] = self.transform(self.horizontal_layers[i])
            
            # convert the vertical graph labels to Float Tensors, this could be necessary if the dataset is not already in Float Tensors
            if self.vertical_graph.y != None:
                self.vertical_graph.y = self.vertical_graph.y.float()
            if self.vertical_graph.edge_attr != None:
                self.vertical_graph.edge_attr = self.vertical_graph.edge_attr.float()
            if self.vertical_graph.x != None:
                self.vertical_graph.x = self.vertical_graph.x.float()
                
            self.vertical_graph = self.transform(self.vertical_graph)

    def print_graph_information(self, save_path=None):
        output = f'\nDataset {self.graph_name}:'
        output += f'\n===================='
        output += f'\nNumber of graphs: {len(self.get_horizontal_layers())}'
        for i in range(len(self.get_horizontal_layers())):
            output += f'\n##### Layer {i}:'
            output += f'\nNumber of nodes in layer {i}: {self.get_horizontal_layers()[i].x.shape[0]}'
            try:
                output += f'\nNumber of node features in layer {i}: {self.get_horizontal_layers()[i].x.shape[1]}'
            except:
                output += f'\nno node features in layer {i}'
            output += f'\nNumber of edges in layer {i}: {self.get_horizontal_layers()[i].edge_index.shape[1]}'
            if (self.get_horizontal_layers()[i].edge_attr != None):
                output += f'\nNumber of edge features in layer {i}: {self.get_horizontal_layers()[i].edge_attr.shape[1]}'
            output += f'\nAverage node degree in layer {i}: {self.get_horizontal_layers()[i].edge_index.shape[1] / self.get_horizontal_layers()[i].x.shape[0]:.2f}'
            output += f'\nHas isolated nodes in layer {i}: {self.get_horizontal_layers()[i].has_isolated_nodes()}'
            output += f'\nHas self-loops in layer {i}: {self.get_horizontal_layers()[i].has_self_loops()}'
            output += f'\nIs undirected in layer {i}: {self.get_horizontal_layers()[i].is_undirected()}'
            output += "\n#####\n"
        output += '\n===================='
        output += f'\nNumber of nodes in vertical graph: {self.get_vertical_graph().x.shape[0]}'
        try:
            output += f'\nNumber of node features in vertical graph: {self.get_vertical_graph().x.shape[1]}'
        except:
            output += f'\nno node features in vertical graph'
        output += f'\nNumber of edges in vertical graph: {self.get_vertical_graph().edge_index.shape[1]}'
        if (self.get_vertical_graph().edge_attr != None):
            output += f'\nNumber of edge features in vertical graph: {self.get_vertical_graph().edge_attr.shape[1]}'
        output += f'\nAverage node degree in vertical graph: {self.get_vertical_graph().edge_index.shape[1] / self.get_vertical_graph().x.shape[0]:.2f}'
        output += f'\nHas isolated nodes in vertical graph: {self.get_vertical_graph().has_isolated_nodes()}'
        output += f'\nHas self-loops in vertical graph: {self.get_vertical_graph().has_self_loops()}'
        output += f'\nIs undirected in vertical graph: {self.get_vertical_graph().is_undirected()}'
        output += '\n===================='
        
        # count the number of edges that target a specific layer
        edge_count = [0] * len(self.get_horizontal_layers())
        # get the number of nodes in each layer
        node_per_layer = [len(self.get_horizontal_layers()[i].x) for i in range(len(self.get_horizontal_layers()))]
        for i in range(len(self.get_horizontal_layers())):
            # count how many nodes are in the layers before the current layer
            prev_node_count = sum(node_per_layer[:i])
            # count how many edges target the current layer, the target should be in range [prev_node_count, prev_node_count + node_per_layer[i]]
            for j in range(self.get_vertical_graph().edge_index.shape[1]):
                if (self.get_vertical_graph().edge_index[1][j] >= prev_node_count and self.get_vertical_graph().edge_index[1][j] < prev_node_count + node_per_layer[i]):
                    edge_count[i] += 1
            output += f'\nNumber of edges in vertical graph that target layer {i}: {edge_count[i]}'
            # define a matrix that stores the number of edges between each layer
            edge_count_between_layers = [[0] * len(self.get_horizontal_layers())] * len(self.get_horizontal_layers())
            for j in range(len(self.get_horizontal_layers())):
                for k in range(self.get_vertical_graph().edge_index.shape[1]):
                    if(self.get_vertical_graph().edge_index[1][k] >= prev_node_count and self.get_vertical_graph().edge_index[1][k] < prev_node_count + node_per_layer[i]):
                        if(self.get_vertical_graph().edge_index[0][k] >= sum(node_per_layer[:j]) and self.get_vertical_graph().edge_index[0][k] < sum(node_per_layer[:j]) + node_per_layer[j]):
                            edge_count_between_layers[i][j] += 1
                output += f'\nNumber of edges in vertical graph that target layer {i} and come from layer {j}:\t {edge_count_between_layers[i][j]}'

        print(output)
        if save_path != None:
            # create the folder on the given path if those don't exist
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            with open(save_path, 'w') as f:
                f.write(output)
    
    def visualize_graph(self):
        plt.figure(figsize=(12,12), dpi=1000, facecolor='w', edgecolor='k')
        for i in range(len(self.get_horizontal_layers())):
            g=to_networkx(self.get_horizontal_layers()[i], to_undirected=True)
            nx.draw(g, with_labels = True, node_size=30,font_size=5, pos=nx.kamada_kawai_layout(g), node_color='r', edge_color='b')
        
        g=to_networkx(self.get_vertical_graph(), to_undirected=True)
        nx.draw(g, with_labels = True, node_size=30,font_size=5, pos=nx.kamada_kawai_layout(g), node_color='r', edge_color='g')

        outpath = "src/mpxgat/visualization/" + self.graph_name + ".png"
        plt.savefig(outpath)
        # clearing the current plot
        plt.clf()
        
    def augment_node_features(self):
        print("augmenting node features...")
        """
        for each graph layer, add new node feature that reflect the node topology in the graph like node degree, node centralities, etc.
        """

        for i in range(len(self.get_horizontal_layers())):
            # out degree (number of outgoing edges, degree calculate for each node in the graph, .view(-1,1) is used to reshape the tensor (put it to column) to be concatenated with the node features tensor)
            self.get_horizontal_layers()[i].x = torch.cat([degree(self.get_horizontal_layers()[i].edge_index[0], num_nodes=self.get_horizontal_layers()[i].x.shape[0]).view(-1,1), self.get_horizontal_layers()[i].x], dim=1)
            # in degree
            self.get_horizontal_layers()[i].x = torch.cat([degree(self.get_horizontal_layers()[i].edge_index[1], num_nodes=self.get_horizontal_layers()[i].x.shape[0]).view(-1,1), self.get_horizontal_layers()[i].x], dim=1)
            
            G = nx.Graph()
            # set the number of nodes in the graph
            G.add_nodes_from(list(range(self.get_horizontal_layers()[i].x.shape[0])))
            G.add_edges_from(self.get_horizontal_layers()[i].cpu().edge_index.T.numpy())

            # node centrality for each horizontal layer
            # centrality = nx.betweenness_centrality(G)
            # centrality_tensor = torch.tensor(list(centrality.values())).view(-1, 1) #! removed because it is too heavy to compute
            # self.get_horizontal_layers()[i].x = torch.cat([centrality_tensor, self.get_horizontal_layers()[i].x], dim=1)
            
            # degree centrality
            degree_centrality = nx.degree_centrality(G)
            degree_centrality_tensor = torch.tensor(list(degree_centrality.values())).view(-1, 1) 
            self.get_horizontal_layers()[i].x = torch.cat([degree_centrality_tensor, self.get_horizontal_layers()[i].x], dim=1)
            
        
        # node degree for vertical graph (out degree, in degree)
        self.get_vertical_graph().x = torch.cat([degree(self.get_vertical_graph().edge_index[0], num_nodes=self.get_vertical_graph().x.shape[0]).view(-1,1), self.get_vertical_graph().x], dim=1)
        self.get_vertical_graph().x = torch.cat([degree(self.get_vertical_graph().edge_index[1], num_nodes=self.get_vertical_graph().x.shape[0]).view(-1,1), self.get_vertical_graph().x], dim=1)
        
        G = nx.Graph()
        G.add_nodes_from(list(range(self.get_vertical_graph().x.shape[0])))
        G.add_edges_from(self.get_vertical_graph().cpu().edge_index.T.numpy())

        #  node centrality for the vertical graph
        # centrality = nx.betweenness_centrality(G)
        # centrality_tensor = torch.tensor(list(centrality.values())).view(-1, 1) #! removed because it is too heavy to compute
        # self.get_vertical_graph().x = torch.cat([centrality_tensor, self.get_vertical_graph().x], dim=1)
        
        # degree centrality
        degree_centrality = nx.degree_centrality(G)
        degree_centrality_tensor = torch.tensor(list(degree_centrality.values())).view(-1, 1)
        self.get_vertical_graph().x = torch.cat([degree_centrality_tensor, self.get_vertical_graph().x], dim=1)


        #! Note for betweenness centrality: this implementation assumes that the graphs are undirected. 
        #! If not, a different centrality measure must be used, e.g. betweenness_centrality_source or betweenness_centrality_target.
        