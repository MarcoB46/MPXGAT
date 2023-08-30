import json
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from mpxgat.multiplex_graph import MultiplexGraph
from tqdm import tqdm
import os.path


class MultiplexSageAdapter:
    """ Read a dataset from the ones used in the MultiplexSage paper and adapt it to the MultiplexGraph class
        the file structure is the following:
        ```
        {
        "directed": false, # tells if the graph is directed or not
        "multigraph": false, # tells if the graph is a multigraph or not
        "graph": {}, # graph attributes
        "nodes": [ # list of nodes
            {
                "id": 0, # node id
                "sheet": 0, # node sheet (layer)
                "val": false, # node validation set membership
                "test": false, # node test set membership
                "marked": false, # node marked
            },
            ...
        ],
        "links": [ # list of links
            {
                "source": 0, # source node id
                "target": 1, # target node id
                "train_removed": false, # link train_removed
                "inter_layer": false, # link inter_layer
            },
            ...
        ]
        }
        ```
        The dataset is composed of different layers, each one representing a different type of interaction between the nodes.
        the horizontal layers are mapped in a list of Data objects.
        the vertical layers is mapped in a single Data object. 
        The structure used is the one defined inside the /src/multiplex_graph.py file.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph : MultiplexGraph = None
    """ MultiplexGraph object to rappresent the Multiple Graph.
    It is composed by a list of Data objects, one for each horizontal layer and
    a single Data object for the vertical layer."""
    num_layers = None
    """ Number of layers (horizontal layers)"""
    horizontal_edges = None
    """ List of edges for each layer (horizontal layer)"""
    horizontal_edges_train_mask = None
    """ List of edges for each layer (horizontal layer) used for training"""
    horizontal_edges_test_mask = None
    """ List of edges for each layer (horizontal layer) used for testing"""
    vertical_edges = None
    """ List of edges for the vertical layer"""
    vertical_edges_train_mask = None
    """ List of edges for the vertical layer used for training"""
    vertical_edges_test_mask = None
    """ List of edges for the vertical layer used for testing"""
    num_nodes_list_horizontal = None
    """ List of number of nodes for each layer (horizontal layer)"""
    num_nodes_vertical = None
    """ Number of nodes for the vertical layer (equal to the total number of nodes on the dataset) """
    data = None
    """ Json object containing the dataset """
    rawData = None
    """ List of unprocessed Data objects read as intermediate step to create the MultiplexGraph object """
    
    def __init__(self):
        pass
    
    
    def loadOrCreateGraph(self, basePath, jsonName, finalGraphName, applyTransforms=False):
        
        torch.device(self.device)
        torch.set_default_tensor_type('torch.FloatTensor')
        # check if the json file is already been converted in a .pt file
        print('####################################################')
        print(f"try to load graph from {basePath+jsonName}.pt file")
        if os.path.isfile(basePath+jsonName+'.pt'):
            self.rawData = torch.load(basePath+jsonName + ".pt")
            self.num_layers = self.rawData['num_layers']
            self.horizontal_edges = self.rawData['horizontal_edges']
            self.horizontal_edges_train_mask = self.rawData['horizontal_edges_train_mask']
            self.horizontal_edges_test_mask = self.rawData['horizontal_edges_test_mask']
            self.vertical_edges = self.rawData['vertical_edges']
            self.vertical_edges_train_mask = self.rawData['vertical_edges_train_mask']
            self.vertical_edges_test_mask = self.rawData['vertical_edges_test_mask']
            self.num_nodes_list_horizontal = self.rawData['num_nodes_list_horizontal']
            self.num_nodes_vertical = self.rawData['num_nodes_vertical']
            self.data = self.rawData['data']
            print("Dataset loaded from .pt file")
        else:
            print(".pt file not found, loading .json file: ", basePath+jsonName + ".json")
            # load the json file
            with open(basePath+jsonName + ".json") as f:
                self.data = json.load(f)
            self.num_layers = self.data['graph']['num_layers']
            
            # store the edges for each horizontal layer
            self.horizontal_edges = []
            self.horizontal_edges_train_mask = []
            self.horizontal_edges_test_mask = []
            print("Storing edges for each layer")
            for i in range(self.num_layers):
                print("Storing edges for layer " + str(i))
                edges = [link for link in self.data["links"] if not link["inter_layer"] and link["source"] in [node["id"] for node in self.data["nodes"] if node["sheet"] == i] and link["target"] in [node["id"] for node in self.data["nodes"] if node["sheet"] == i]]
                # using edges list create a train mask (using the "train_removed" attribute)
                self.horizontal_edges_train_mask.append([not link["train_removed"] for link in edges])
                self.horizontal_edges_test_mask.append([link["train_removed"] for link in edges])
                self.horizontal_edges.append(edges)
                print("edges for layer " + str(i) + " stored : ", len(edges))
            
            # store the number of nodes for each horizontal layer
            self.num_nodes_list_horizontal = [len([node for node in self.data["nodes"] if node["sheet"] == i]) for i in range(self.num_layers)]

            # correct the edge list for each layer (horizontal layer), subtracting the node contained in the previous layers to the source and target node
            print("Correcting edges for each layer")
            for i in range(self.num_layers):
                print("Correcting edges for layer " + str(i))
                for link in self.horizontal_edges[i]:
                    link["source"] -= sum(self.num_nodes_list_horizontal[:i])
                    link["target"] -= sum(self.num_nodes_list_horizontal[:i])
                
            # store the edges for the vertical layer
            self.vertical_edges = [link for link in self.data["links"] if link["inter_layer"]]
            self.vertical_edges_train_mask = [not link["train_removed"] for link in self.vertical_edges]
            self.vertical_edges_test_mask = [link["train_removed"] for link in self.vertical_edges]
            
            # store the number of nodes for the vertical layer
            self.num_nodes_vertical = len(self.data['nodes'])
            
            # save the extracted data in a .pt file
            print("Saving dataset in .pt file", basePath+jsonName + ".pt")
            self.rawData = {
                'num_layers': self.num_layers,
                'horizontal_edges': self.horizontal_edges,
                'horizontal_edges_train_mask': self.horizontal_edges_train_mask,
                'horizontal_edges_test_mask': self.horizontal_edges_test_mask,
                'vertical_edges': self.vertical_edges,
                'vertical_edges_train_mask': self.vertical_edges_train_mask,
                'vertical_edges_test_mask': self.vertical_edges_test_mask,
                'num_nodes_list_horizontal': self.num_nodes_list_horizontal,
                'num_nodes_vertical': self.num_nodes_vertical,
                'data': self.data,
                }
            
            torch.save(self.rawData, basePath+jsonName + ".pt")
            print(f"Dataset saved in {basePath+jsonName}.pt file")
            
        print('####################################################') 
        # try to load the graph from the .pt file
        if os.path.isfile(basePath+finalGraphName+'.pt'):
            self.graph = MultiplexGraph.load(basePath+finalGraphName + ".pt")
            print("Graph loaded from .pt file")
        else:
            print("Graph not found, creating graph")
            self.horizontalGraphs = []
            self.horizontal_nodes_one_hot = []
            # create the edge_index tensor for each horizontal layer
            for i in range(self.num_layers):
                print("Creating edge_index tensor for layer " + str(i))
                self.horizontalGraphs.append(
                    torch.tensor([[link["source"], link["target"]] for link in self.horizontal_edges[i]], dtype=torch.long).t().contiguous()
                )
                # convert the horizontal_train_mask and horizontal_test_mask lists to tensors
                self.horizontal_edges_train_mask[i] = torch.tensor(self.horizontal_edges_train_mask[i])
                self.horizontal_edges_test_mask[i] = torch.tensor(self.horizontal_edges_test_mask[i])
                # create a one hot encoding for each node in the horizontal layer
                self.horizontal_nodes_one_hot.append(torch.eye(self.num_nodes_list_horizontal[i]))
                print("edge_index tensor for layer " + str(i) + " created")
            
            # create the edge_index tensor for the vertical layer
            print("Creating edge_index tensor for vertical layer")
            self.verticalGraph = torch.tensor([[link["source"], link["target"]] for link in self.vertical_edges], dtype=torch.long).t().contiguous()
            # convert the vertical_train_mask and vertical_test_mask lists to tensors
            self.vertical_edges_train_mask = torch.tensor(self.vertical_edges_train_mask)
            self.vertical_edges_test_mask = torch.tensor(self.vertical_edges_test_mask)
            # create a one hot encoding for each node in the vertical layer
            self.vertical_nodes_one_hot = torch.eye(self.num_nodes_vertical)
            
            
            
            # create the Data object for each horizontal layer
            horizontalDataList = []
            for i in range(self.num_layers):
                print("Creating Data object for layer " + str(i))
                horizontalDataList.append(Data(
                    x=self.horizontal_nodes_one_hot[i],
                    edge_index=self.horizontalGraphs[i],
                    edge_attr=None,
                    y=None,
                    train_mask=self.horizontal_edges_train_mask[i],
                    test_mask=self.horizontal_edges_test_mask[i]
                ))
                print("Data object for layer " + str(i) + " created")
            
            # create the Data object for the vertical layer
            print("Creating Data object for vertical layer")
            verticalData = Data(
                x=self.vertical_nodes_one_hot,
                edge_index=self.verticalGraph,
                edge_attr=None,
                y=None,
                train_mask=self.vertical_edges_train_mask,
                test_mask=self.vertical_edges_test_mask
            )
            
            # create the MultiplexGraph object
            print("Creating MultiplexGraph object")
            
            # create the MultiplexGraph object, using the Data objects created before
            self.graph = MultiplexGraph(
                horizontal_layers=horizontalDataList,
                vertical_graph=verticalData,
                graphName=finalGraphName
            )

            print("MultiplexGraph object created")
            # save the graph in a .pt file
            self.graph.save(basePath+finalGraphName + ".pt")
        
        if (applyTransforms):
            self.applyDefaultTransforms(self.graph)
        
        return self.graph
    
    def applyDefaultTransforms(self, graph):
        transform = T.Compose([T.ToUndirected(), T.AddSelfLoops(), T.ToDevice(device=self.device)])
        graph.set_transform(transform)

    def get_graph(self):
        if self.graph is None:
            raise ValueError("Graph has not been initialized.")
        return self.graph
    
    def set_graph(self, graph):
        self.graph = graph
        
    def checkIfMultiplexDatasetExists(self, path):
        if os.path.isfile(path) == True:
            return True
        else:
            return False