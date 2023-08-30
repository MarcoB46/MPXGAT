
import torch_geometric.transforms as T
from tqdm import tqdm
from torch_geometric.data import Data
from mpxgat.multiplex_graph import MultiplexGraph
import os.path
from mpxgat.dataset_adapters.multiplexSageAdapter import MultiplexSageAdapter
from mpxgat.customRandomLinkSplit import CustomRandomLinkSplit
import torch
from torch_geometric.utils import add_random_edge, erdos_renyi_graph, barabasi_albert_graph
import random
from copy import deepcopy

TRAIN = 0
VAL = 1
TEST = 2

class DatasetUtils:
    multiplexGraph: MultiplexGraph
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, datasetName=None, datasetJsonName=None, baseDataPath='Graphs/', printDatasetInfo=False, applyTransforms=False, augmentNodeFeatures=False, useRandomGraph = False):
        """
        Initialize the dataset utils class.
        Parameters are:
        - datasetName: the name of the dataset to load or create, this is equal to name of the file without the extension
        - datasetJsonName: the name of the json file containing the dataset information
        - baseDataPath: the base path where the dataset is stored
        - printDatasetInfo: whether to print the dataset information
        - applyTransforms: whether to apply the default transforms to the dataset
        - augmentNodeFeatures: whether to augment the node features of the dataset, if true the node features will be augmented with the node degree and degree centrality
        - useRandomGraph: whether to use a random graph instead of a real dataset
        """    
        self.datasetName = datasetName
        self.baseDataPath = baseDataPath
        self.printDatasetInfo = printDatasetInfo
        self.applyTransforms = applyTransforms
        self.datasetAdapter = MultiplexSageAdapter()
        self.augmentNodeFeatures = augmentNodeFeatures
        self.setApplyTransforms(applyTransforms)
        self.useRandomGraph = useRandomGraph
        
        if self.useRandomGraph == True:
            return

        # check if a dataset is already generated inside the baseDataPath folder
        if os.path.isfile(baseDataPath+datasetName+'.pt') == True:
            print("Dataset already generated, loading...")
            self.multiplexGraph = MultiplexGraph()
            self.multiplexGraph = self.multiplexGraph.load(path=baseDataPath+datasetName+'.pt')

            if applyTransforms == True:
                self.datasetAdapter.applyDefaultTransforms(self.multiplexGraph)

        else:
            print(F"Dataset not found, generating... {baseDataPath+datasetName+'.pt'}")
            # create the dataset from the adapter
            self.multiplexGraph = self.datasetAdapter.loadOrCreateGraph(basePath=baseDataPath, finalGraphName=datasetName, jsonName=datasetJsonName, applyTransforms=applyTransforms)
            
        if self.printDatasetInfo == True:
            # print dataset info
            self.multiplexGraph.print_graph_information()

        if augmentNodeFeatures == True:
            self.multiplexGraph.augment_node_features()

        self.multiplexGraph.set_transform(transform=self.transform)

    def setApplyTransforms(self, applyTransforms):
        self.applyTransforms = applyTransforms
        if applyTransforms == True:
            self.transform = T.Compose([
                T.NormalizeFeatures(),
                T.ToDevice(self.device),
                CustomRandomLinkSplit(num_val=0, num_test=0.3, is_undirected=True,
                                add_negative_train_samples=False),
            ])
        else:
            self.transform = T.Compose([
                T.NormalizeFeatures(),
                T.ToDevice(self.device)
            ])

    def getHorizontalData(self):
        """
        Returns the horizontal data list representing the layers of a multiplex graph.

        If `transforms` is `True`, the returned data will be the train and test data lists. If `transforms` is `False`,
        the returned data will be the same as used in the MultiplexSAGE paper.

        Args:
            transforms (bool, optional): Whether to apply transforms to the data. Defaults to `False`.

        Returns:
            list: The horizontal data list representing the layers of the multiplex graph. If `transforms` is `True`,
            the returned data will be the train and test data lists.
        """
        if self.applyTransforms == False:
            return self.multiplexGraph.get_horizontal_layers()
        else:
            self.horizontal_train_data_list = []
            self.horizontal_test_data_list = []
            # self.horizontal_val_data_list = []
            for i in range(len(self.multiplexGraph.get_horizontal_layers())):
                self.horizontal_train_data_list.append(self.multiplexGraph.get_horizontal_layers()[i][0])
                self.horizontal_test_data_list.append(self.multiplexGraph.get_horizontal_layers()[i][1])
                # self.horizontal_val_data_list.append(self.multiplexGraph.get_horizontal_layers()[i][2])

            return self.horizontal_train_data_list, self.horizontal_test_data_list
    
    def getVerticalData(self):
        if self.applyTransforms == False:
            return self.multiplexGraph.get_vertical_graph()
        else :
            # get training, validation and test data for the vertical custom gat layer
            self.vertical_train_data = self.multiplexGraph.get_vertical_graph()[0]
            self.vertical_test_data = self.multiplexGraph.get_vertical_graph()[1]
            # self.vertical_test_data = self.multiplexGraph.get_vertical_graph()[2]
            return self.vertical_train_data, self.vertical_test_data
    
    def getDataset(self):
        return self.multiplexGraph
    
    def getTopKLayers(self, k, customOrder=None):
        """
        Return the top k layers per number of nodes in the graph or in a custom order if provided.
        """
        # sort the horizontal layers based on the number of nodes in each layer or a custom order if provided
        sortedHorizontalLayers = [None] * len(self.getDataset().get_horizontal_layers())
        if customOrder == None:
            sortedHorizontalLayers = sorted(self.getDataset().get_horizontal_layers(), key=lambda x: x[0].num_nodes, reverse=True)
        else:
            for i,j in customOrder:
                sortedHorizontalLayers[j] = self.getDataset().get_horizontal_layers()[i]

        # get the number of nodes in each layer
        num_nodes_per_layer = torch.tensor([[i, self.getDataset().get_horizontal_layers()[i].num_nodes] for i in range(len(self.getDataset().get_horizontal_layers()))])

        # for each layer get the range of node ids in each (for example, if the first layer has 10 nodes, the range will be 0-9)
        node_ids_per_layer = torch.tensor([[i,
            (i == 0 and 0 or sum([self.getDataset().get_horizontal_layers()[j].num_nodes for j in range(i)])),
            self.getDataset().get_horizontal_layers()[i].num_nodes - 1 + sum([self.getDataset().get_horizontal_layers()[j].num_nodes for j in range(i)]),
        ] for i in range(len(self.getDataset().get_horizontal_layers()))])
        
        
        ordered_num_nodes_per_layer = [None] * len(sortedHorizontalLayers)
        if customOrder == None:
            ordered_num_nodes_per_layer = num_nodes_per_layer[num_nodes_per_layer[:, 1].argsort(descending=True)]
        else:
            for i,j in customOrder:
                ordered_num_nodes_per_layer[j] = num_nodes_per_layer[i]
            ordered_num_nodes_per_layer = torch.stack(ordered_num_nodes_per_layer)
    

        ordered_node_ids_per_layer = torch.tensor([[
            ordered_num_nodes_per_layer[i][0], # layer id
            (i == 0 and 0 or sum([ordered_num_nodes_per_layer[j][1] for j in range(i)])), # start of the range
            ordered_num_nodes_per_layer[i][1] - 1 + sum([ordered_num_nodes_per_layer[j][1] for j in range(i)]),
        ] for i in range(len(sortedHorizontalLayers))])

        # map each possible node id to the relative ordered node id
        mapping = []
        for i in range(len(ordered_num_nodes_per_layer)):
            layer_id = ordered_num_nodes_per_layer[i][0]
            # get the unordered node ids range in the layer
            range_unordered = node_ids_per_layer[node_ids_per_layer[:, 0] == layer_id][:, 1:]
            range_ordered = ordered_node_ids_per_layer[ordered_node_ids_per_layer[:, 0] == layer_id][:, 1:]

            mapping.append(torch.tensor(list(zip(range(range_unordered[0][0].item(), range_unordered[0][1].item()+1), range(range_ordered[0][0].item(), range_ordered[0][1].item()+1)))))
        # merge the mappings
        mapping = torch.cat(mapping)

        # cycle through the edge_index of the vertical graph and map the node ids
        pbar = tqdm(total=self.getDataset().get_vertical_graph().edge_index.shape[1], position=0, leave=True, desc="Mapping node ids")
        new_edge_index = torch.zeros_like(self.getDataset().get_vertical_graph().edge_index)
        for i in range(self.getDataset().get_vertical_graph().edge_index.shape[1]):
            new_edge_index[0][i] = mapping[mapping[:, 0] == self.getDataset().get_vertical_graph().edge_index[0][i].item()][0][1]
            
            new_edge_index[1][i] = mapping[mapping[:, 0] == self.getDataset().get_vertical_graph().edge_index[1][i].item()][0][1]
            
            pbar.update(1)
            
        pbar.close()
        
        # order the node features of the vertical graph based on the mapping
        pbar = tqdm(total=len(mapping), position=0, leave=True, desc="Ordering node features")
        ordered_node_features = torch.zeros_like(self.getDataset().get_vertical_graph().x)
        for i in range(len(mapping)):
            ordered_node_features[mapping[i][1]] = self.getDataset().get_vertical_graph().x[mapping[i][0]]
            pbar.update(1)
        pbar.close()
        
        
        #get the number of nodes in the first k layers
        num_nodes_in_top_k_layers = ordered_num_nodes_per_layer[:k][:, 1]
        
        new_node_features = ordered_node_features[:num_nodes_in_top_k_layers.sum(), :]

        # remove from the edge_index the edges that are not in the top k layers (the ones that have a node id greater than the number of nodes in the top k layers)
        new_edge_index = new_edge_index[:, (new_edge_index[0] < num_nodes_in_top_k_layers.sum()) & (new_edge_index[1] < num_nodes_in_top_k_layers.sum())]
        
        # create the new graph
        sortedVerticalLayer = Data(x=new_node_features, edge_index=new_edge_index, num_nodes=num_nodes_in_top_k_layers.sum())
        
        # get the top k horizontal layers
        sortedHorizontalLayers = sortedHorizontalLayers[:k]
        
        orderedMPXGraph = MultiplexGraph(
            horizontal_layers=sortedHorizontalLayers,
            vertical_graph=sortedVerticalLayer,
            graphName=self.getDataset().graph_name+"_top_"+str(k),
        )
        
        if self.augmentNodeFeatures is True:
            orderedMPXGraph.augment_node_features()
            
        if self.printDatasetInfo is True:
            print(f"#######\t\tgraph top {k}\t\t#######")
            orderedMPXGraph.print_graph_information(save_path=self.baseDataPath+"topKLayers/"+orderedMPXGraph.graph_name+"-"+str(customOrder).replace(" ", "")+".txt")
        
        if self.applyTransforms is True:
            self.datasetAdapter.applyDefaultTransforms(orderedMPXGraph)
            orderedMPXGraph.set_transform(self.transform)
        
        return orderedMPXGraph
    
    def getRewiringProbabilityPerLayerDensity(self, graph: MultiplexGraph, rewiringRegularization: float = 0.8):
        """
        Return a list of probability, ```[0,1]```, for each layer in the graph.
        Layers with low density will have an exponentially higher probability of rewiring.
        """
        
        # get the edge density of each layer
        edge_density_per_layer = torch.tensor([[i, graph.get_horizontal_layers()[i].num_edges / (graph.get_horizontal_layers()[i].num_nodes)] for i in range(len(graph.get_horizontal_layers()))])
        max_edge_density = edge_density_per_layer[:, 1].max()
        min_edge_density = edge_density_per_layer[:, 1].min()
        edge_density_per_layer[:, 1] = (max_edge_density - edge_density_per_layer[:, 1]) / (max_edge_density - min_edge_density)
        
        edge_density_per_layer = edge_density_per_layer[:, 1] * rewiringRegularization
        
        return edge_density_per_layer
    
    def applyRewiringProbabilityPerLayerDensity(self, rewiringRegularization: float = 0.8):
        """
        Apply the rewiring probability per layer density to the graph by adding edges per layer with a given probability.
        """
        # get the original Graph
        graph: MultiplexGraph = self.multiplexGraph.load(path=self.baseDataPath+self.datasetName+'.pt')
        
        # applying the default transform
        self.datasetAdapter.applyDefaultTransforms(graph)
        
        
        # get the rewiring probability per layer density
        rewiring_probability_per_layer_density = self.getRewiringProbabilityPerLayerDensity(graph, rewiringRegularization=rewiringRegularization)
        
        graph_h = graph.get_horizontal_layers()
        for layer in range (len(rewiring_probability_per_layer_density)):
            rewiring_probability = rewiring_probability_per_layer_density[layer].item()
            rewiring_probability = round(rewiring_probability, 2)
            print("Rewiring layer "+str(layer))
            # add edges with a probability of rewiring_probability_per_layer_density[layer][1]
            edge_index, _ = add_random_edge(graph_h[layer].edge_index, rewiring_probability, True, graph_h[layer].num_nodes) # _ is the list of added edges, we don't need it
            graph_h[layer].edge_index = edge_index
        
        graph.set_transform(transform=self.transform)       
        return graph

    def generateRandomMultiplexGraph(self, numNodes: int = 1000, p_h: int|list=0.5, p_v: int|list=0.1, numLayers: int = 10, isUndirected: bool = True, lenNodeFeature: int = 100, graphNamePrefix: str = "randomMultiplexGraph"):
        """
        Generate or load a random multiplex graph with the given parameters.
        """
        # check if the graph already exists
        if os.path.exists(self.baseDataPath+graphNamePrefix + f"_#{numNodes}_ph{p_h}_pb{p_v}_K{numLayers}.pt"):
            print("Loading the graph")
            self.multiplexGraph = MultiplexGraph()
            self.multiplexGraph = self.multiplexGraph.load(path=self.baseDataPath+graphNamePrefix + f"_#{numNodes}_ph{p_h}_pb{p_v}_K{numLayers}.pt")
        else: # otherwise generate the graph
            print("Generating the graph")
            graphName = graphNamePrefix +f"_#{numNodes}_ph{p_h}_pb{p_v}_K{numLayers}"
            
            # generate the horizontal layers
            horizontal_layers_edge_index = []

            if type(p_h) is list and len(p_h) != numLayers:
                raise Exception("The length of the list of probabilities for the horizontal layers must be equal to the number of layers")

            for i in range(numLayers):
                # horizontal_layers_edge_index.append(erdos_renyi_graph(num_nodes=numNodes, edge_prob=p_h[i] if type(p_h) is list else p_h, directed=(not isUndirected)))
                horizontal_layers_edge_index.append(barabasi_albert_graph(num_edges=int(numNodes * (p_h[i] if type(p_h) is list else p_h)), num_nodes=numNodes))
                    
            # generate the horizontal layers
            horizontal_layers = []
            for i in range(len(horizontal_layers_edge_index)):
                # generate a list of node features, first generate a random number between 1 and numLayers, then generate a matrix of shape numNodes x numNodes with random values between 0 and 1 and multiply it by the random number
                node_features = torch.rand((numNodes, lenNodeFeature)) * (i+1) #random.randint(1, numLayers)
                node_features = node_features.to(torch.float)
                node_features = node_features.to(self.device)
                horizontal_layers_edge_index[i] = torch.tensor(horizontal_layers_edge_index[i], dtype=torch.long).to(self.device)
                horizontal_layers.append(Data(edge_index=horizontal_layers_edge_index[i], num_nodes=numNodes, x=node_features, graphName=graphName+"_horizontal_layer_"+str(i)))
            
            # generate the vertical layer
            
            # generate the vertical edges (source and target must be in different layers)
            vertical_edges = torch.tensor([], dtype=torch.long).to(self.device)
            for i in range(numLayers-1):
                if type(p_v) is list and len(p_v) != numLayers:
                    raise Exception("The length of the list of probabilities for the vertical layers must be equal to the number of layers")
                # add to each source and target the number of nodes in the previous layers
                vertical_edges_temp = erdos_renyi_graph(num_nodes=numNodes, edge_prob=p_v[i] if type(p_v) is list else p_v, directed=(not isUndirected)).to(self.device)
                vertical_edges_temp[0] += numNodes * i
                vertical_edges_temp[1] += numNodes * (i+1)
                vertical_edges = torch.cat((vertical_edges, vertical_edges_temp), dim=1)
            
            
            # concatenate the horizontal node features to create the vertical node features
            vertical_node_features = torch.cat([horizontal_layers[i].x for i in range(len(horizontal_layers))], dim=0)
            vertical_layer = Data(edge_index=vertical_edges, num_nodes=numNodes * numLayers, x=vertical_node_features, graphName=graphName+"_vertical_layer")
            
            # create the multiplex graph
            randomMultiplexGraph = MultiplexGraph(
                horizontal_layers=horizontal_layers,
                vertical_graph=vertical_layer,
                graphName=graphName
            )
            
            if self.applyTransforms is True:
                self.datasetAdapter.applyDefaultTransforms(randomMultiplexGraph)
                randomMultiplexGraph.set_transform(self.transform)
            
            # save the graph
            randomMultiplexGraph.save(path=self.baseDataPath+graphName+'.pt')
            self.multiplexGraph = randomMultiplexGraph
        
        return self.multiplexGraph
    
    def getTopKLayersFromRandomGraph(self, graph_temp: MultiplexGraph, k: int = 2):
        graph = deepcopy(graph_temp)
        # get the first k layers
        horizontal_layers = graph.get_horizontal_layers()
        horizontal_layers = horizontal_layers[:k]
        
        num_nodes_per_layer = horizontal_layers[0].num_nodes
        # remove the edges from the vertical layer that are not in the first k layers
        vertical_layer = graph.get_vertical_graph()
        vertical_layer_edge_index = vertical_layer.edge_index
        vertical_layer_edge_index = vertical_layer_edge_index[:, (vertical_layer_edge_index[0] < k*num_nodes_per_layer) & (vertical_layer_edge_index[1] < k*num_nodes_per_layer)]
        vertical_layer.x = vertical_layer.x[:k*num_nodes_per_layer]
        # create the new graph
        newGraph = MultiplexGraph(
            horizontal_layers=horizontal_layers,
            vertical_graph=Data(edge_index=vertical_layer_edge_index, num_nodes=num_nodes_per_layer*k, x=vertical_layer.x),
            graphName=graph.graph_name+"_top_"+str(k)
        )
        
        if self.augmentNodeFeatures is True:
            newGraph.augment_node_features()
            
        if self.printDatasetInfo is True:
            print(f"#######\t\tgraph top {k}\t\t#######")
            newGraph.print_graph_information()
        
        if self.applyTransforms is True:
            self.datasetAdapter.applyDefaultTransforms(newGraph)
            newGraph.set_transform(self.transform)

        return newGraph