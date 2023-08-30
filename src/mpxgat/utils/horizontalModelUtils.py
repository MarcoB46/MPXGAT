import datetime
from mpxgat.MPXGAT_H import MPXGAT_H
import json
import torch
from torch_geometric.utils import negative_sampling
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,  roc_auc_score, classification_report

import os.path
from mpxgat.utils.datasetUtils import DatasetUtils
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.EarlyStopper import EarlyStopper
from mpxgat.multiplex_graph import MultiplexGraph
import copy


class HorizontalModelUtils(object):
    model: MPXGAT_H
    """ model to be trained, this is the submodel that will be trained on each horizontal layer """
    
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """ device to be used for training and inference """
    
    datasetUtils: DatasetUtils
    """ utils class to load the dataset """
    
    datasetCut: bool
    """ flag to indicate if the train/test split to be used is the one provided by the dataset or a random one """
    
    earlyStopingPatience: int = 5
    """ number of epochs to wait before stopping the training if the validation loss does not improve """
    
    customGraph: MultiplexGraph = None
    
    datasetParametersList = [
        { #drosophila
            'hiddenChannels': [128], 'lr':[ 0.0005], 'numEpochs':[ 1500], 'numHeads': [3], 'outChannels':[ 128], 'weight_decay': [0.0005]
        },
        { #arxiv            
            'hiddenChannels': [256], 'lr': [0.0005], 'numEpochs': [1500], 'numHeads': [4], 'outChannels': [128], 'weight_decay':[ 5e-05]
        },
        { #FFYTTW            
            'hiddenChannels': [128], 'lr': [1e-05], 'numEpochs': [1500], 'numHeads':[5], 'outChannels': [128], 'weight_decay': [0.05]
        },
        { #random
            'hiddenChannels': [256], 'lr': [1e-05], 'numEpochs': [1500], 'numHeads': [5], 'outChannels': [128], 'weight_decay': [5e-05]
        },
    ]
    
    def print(self, *args):
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " + self.prefixModelName + self.modelName + " :: ", *args)
    
    def __init__(self, modelBasePath, modelName, datasetUtils: DatasetUtils = None, prefixModelName='', datasetKey=None, subfolder=None, customGraph: MultiplexGraph=None, topKGraphKey=None, datasetCut = True):
        self.dataseKey = datasetKey
        self.parameterGrid = self.datasetParametersList[datasetKey]
        self.modelBasePath = modelBasePath
        self.modelName = modelName
        self.prefixModelName = prefixModelName
        self.subfolder = subfolder
        self.customGraph = customGraph
        self.topKGraphKey = topKGraphKey
        self.datasetCut = datasetCut

        self.datasetUtils = datasetUtils
        self.trainDataList = []
        self.testDataList = []
        if self.topKGraphKey != None:
            self.dataList = self.customGraph.get_horizontal_layers()
            for i in range(len(self.dataList)):
                self.trainDataList.append(self.dataList[i][0])
                self.testDataList.append(self.dataList[i][1])
        else:
            if self.datasetCut == True:
                self.dataList = datasetUtils.getHorizontalData()
            else:
                self.trainDataList, self.testDataList = datasetUtils.getHorizontalData()

        self.initTrainTestData()

        self.val_auc = self.test_auc = self.loss = self.epoch = 0
        
        
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # print informations about the train and test datasets
        self.print_dataset_info(self.trainDataList, 'train')
        self.print_dataset_info(self.testDataList, 'test')
        
    def initTrainTestData(self):
        if self.datasetCut == True:
            for i in range(len(self.dataList)):
                self.trainDataList.append(copy.deepcopy(self.dataList[i]))
                self.trainDataList[i].edge_index = self.trainDataList[i].edge_index[:, self.dataList[i].train_mask]
                self.trainDataList[i].edge_label_index = copy.deepcopy(self.trainDataList[i].edge_index)
                self.trainDataList[i].edge_label = torch.ones(self.trainDataList[i].edge_label_index.shape[1])
                neg_edge_index = negative_sampling(edge_index=self.trainDataList[i].edge_index, num_nodes=self.trainDataList[i].num_nodes, num_neg_samples=self.trainDataList[i].num_edges)
                self.trainDataList[i].edge_label_index = torch.cat([self.trainDataList[i].edge_label_index, neg_edge_index], dim=-1).to(self.device)
                self.trainDataList[i].edge_label = torch.cat([self.trainDataList[i].edge_label, torch.zeros(neg_edge_index.shape[1])], dim=-1).to(self.device)

                self.testDataList.append(copy.deepcopy(self.dataList[i]))
                self.testDataList[i].edge_index = self.testDataList[i].edge_index[:, self.dataList[i].test_mask]
                self.testDataList[i].edge_label_index = copy.deepcopy(self.testDataList[i].edge_index)
                self.testDataList[i].edge_label = torch.ones(self.testDataList[i].edge_label_index.shape[1])
                neg_edge_index = negative_sampling(edge_index=self.testDataList[i].edge_index, num_nodes=self.testDataList[i].num_nodes, num_neg_samples=self.testDataList[i].num_edges)
                self.testDataList[i].edge_label_index = torch.cat([self.testDataList[i].edge_label_index, neg_edge_index], dim=-1).to(self.device)
                self.testDataList[i].edge_label = torch.cat([self.testDataList[i].edge_label, torch.zeros(neg_edge_index.shape[1])], dim=-1).to(self.device)
        else:
            return
        
    def print_dataset_info(self, datasetList, prefix):
        print('\n\n\n====================')
        print(f'{prefix} - Number of graphs: {len(datasetList)}')
        for i in range(len(datasetList)):
            print(f'{prefix} ##### Layer {i}:')
            print(f'{prefix} Number of nodes in layer {i}: {datasetList[i].x.shape[0]}')
            try:
                print(f'{prefix} Number of node features in layer {i}: {datasetList[i].x.shape[1]}')
            except:
                print(f'{prefix} no node features in layer {i}')
            print(f'{prefix} Number of edges in layer {i}: {datasetList[i].edge_index.shape[1]}')
            if (datasetList[i].edge_attr != None):
                print(f'{prefix} Number of edge features in layer {i}: {datasetList[i].edge_attr.shape[1]}')
            print(f'{prefix} Average node degree in layer {i}: {datasetList[i].edge_index.shape[1] / datasetList[i].x.shape[0]:.2f}')
            print(f'{prefix} Has isolated nodes in layer {i}: {datasetList[i].has_isolated_nodes()}')
            print(f'{prefix} Has self-loops in layer {i}: {datasetList[i].has_self_loops()}')
            print(f'{prefix} Is undirected in layer {i}: {datasetList[i].is_undirected()}')
            print("#####\n")
        print('====================\n\n\n')

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        # get the list of the encoding of each layer of the multiplex graph
        z_H = self.model.forward(horizontal_graphs=self.trainDataList)
        # performing a negative sampling for every layer of the multiplex graph, for every trainign epoch
        loss_list = [] # list of the loss for each layer of the multiplex graph
        for i in range(len(z_H)):
            train_data = self.trainDataList[i]
            if self.datasetCut == True:
                # calculate the output of the current layer (this is the list of embeddings for each node of the current layer)
                out = self.model.decode(z_H[i], train_data.edge_label_index).view(-1)
                # calculate the loss for the current layer
                loss = self.criterion(out, train_data.edge_label)
            else:
                neg_edge_index = negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes, num_neg_samples=train_data.num_edges)
                # neg_edge_index_list.append(negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,num_neg_samples=train_data.edge_label_index.size(1), method='sparse'))
                # concatenate the negative sampled edges to the positive ones for the current layer
                edge_label_index = torch.cat(
                    [train_data.edge_label_index, neg_edge_index],
                    dim=-1,
                )
                # concatenate the negative sampled labels to the positive ones for the current layer
                edge_label = torch.cat([
                    train_data.edge_label,
                    train_data.edge_label.new_zeros(neg_edge_index.size(1))
                ], dim=0)
                # calculate the output of the current layer (this is the list of embeddings for each node of the current layer)
                out = self.model.decode(z_H[i], edge_label_index).view(-1)
                # calculate the loss for the current layer
                loss = self.criterion(out, edge_label)
            loss_list.append(loss)
            # propagate the loss for the current layer
            loss.backward()
        # update the weights of the model
        self.optimizer.step()
        
        # compute the average loss for the multiplex graph
        loss = sum(loss_list) / len(loss_list)
        return loss

    @torch.no_grad()
    def getValidationLoss(self):
        # get the list of the encoding of each layer of the multiplex graph
        z_H = self.model.forward(horizontal_graphs=self.testDataList)
        # performing a negative sampling for every layer of the multiplex graph, for every trainign epoch
        loss_list = [] # list of the loss for each layer of the multiplex graph
        for i in range(len(z_H)):
            train_data = self.testDataList[i]
            
            if train_data.edge_label_index.size(1) == 0:
                continue
            
            if self.datasetCut == True:
                # calculate the output of the current layer (this is the list of embeddings for each node of the current layer)
                out = self.model.decode(z_H[i], train_data.edge_label_index).view(-1)
                # calculate the loss for the current layer
                loss = self.criterion(out, train_data.edge_label)
            else:
                neg_edge_index = negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes, num_neg_samples=train_data.num_edges)
                # neg_edge_index_list.append(negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,num_neg_samples=train_data.edge_label_index.size(1), method='sparse'))
                # concatenate the negative sampled edges to the positive ones for the current layer
                edge_label_index = torch.cat(
                    [train_data.edge_label_index, neg_edge_index],
                    dim=-1,
                )
                # concatenate the negative sampled labels to the positive ones for the current layer
                edge_label = torch.cat([
                    train_data.edge_label,
                    train_data.edge_label.new_zeros(neg_edge_index.size(1))
                ], dim=0)
                # calculate the output of the current layer (this is the list of embeddings for each node of the current layer)
                out = self.model.decode(z_H[i], edge_label_index).view(-1)
                # calculate the loss for the current layer
                loss = self.criterion(out, edge_label)

            loss_list.append(loss)
        
        # compute the average loss for the multiplex graph
        loss = sum(loss_list) / len(loss_list)
        return loss

    @torch.no_grad()
    def test(self, dataList, printStatistics=False):
        z_H = self.model.forward(dataList)
        roc_list = []
        statistics_list = []
        for i in range(len(z_H)):
            data = dataList[i]
            if data.edge_label_index.size(1) == 0:
                continue
            # the sigmoid function is applied to the output of the model in order to obtain a value between 0 and 1 (probability of the edge to be positive)
            out = self.model.decode(z_H[i], data.edge_label_index).view(-1).sigmoid()
            roc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
            roc_list.append(roc)
            if printStatistics == True:
                threshold = 0.6 # TODO: calculate the optimal threshold like in the singleLayerUtils class.
                out_binary = (out > threshold).float()  # Convert to binary values using a threshold
                # Calculate confusion matrix
                cm = confusion_matrix(data.edge_label.cpu().numpy(), out_binary.cpu().numpy())
                self.print("Confusion matrix:")
                self.print(cm)

                # Calculate precision, recall, and F1 score
                precision = precision_score(data.edge_label.cpu().numpy(), out_binary.cpu().numpy())
                recall = recall_score(data.edge_label.cpu().numpy(), out_binary.cpu().numpy())
                accuracy = accuracy_score(data.edge_label.cpu().numpy(), out_binary.cpu().numpy())
                f1 = f1_score(data.edge_label.cpu().numpy(), out_binary.cpu().numpy())
                y_true = data.edge_label.cpu().numpy()
                y_pred = out_binary.cpu().numpy()
                
                labels = [0, 1]
                target_names = ['Absent', 'Present']
                class_report = classification_report(y_true, y_pred, target_names=target_names, labels=labels, output_dict=True)
                self.print("roc:", roc)
                self.print("Precision:", precision)
                self.print("Recall:", recall)
                self.print("F1 score:", f1)
                self.print("Accuracy:", accuracy)
                
                statistics = {
                    'layer': i,
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "f1": f1,
                    "roc": roc,
                    "confusion_matrix": cm.tolist(),
                    "classification_report": class_report,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # add timestamp to the statistics
                }
                statistics_list.append(statistics)
        if printStatistics == True:
            statistics_list.append({
                'avg_roc': sum(roc_list) / len(roc_list),
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # add timestamp to the statistics
            })
            
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_statistics.json'
            if self.topKGraphKey != None:
                path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_statistics_top-'+str(self.topKGraphKey)+'.json'
            
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            with open(path, 'a') as outfile:
                json.dump(statistics_list, outfile)
                outfile.write('\n')
                
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_avg_roc.txt'
            if self.topKGraphKey != None:
                path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_avg_roc_top-'+str(self.topKGraphKey)+'.txt'
            
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            with open(path, 'a') as outfile:
                json.dump((sum(roc_list) / len(roc_list)), outfile)
                outfile.write('\n')
                
                
        return sum(roc_list) / len(roc_list), roc_list
    
    def checkIfModelExists(self):
        if self.topKGraphKey != None and os.path.isfile(self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_model_top-'+str(self.topKGraphKey)+'.pt') == True:
            return True
        
        if os.path.isfile(self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_model.pt') == True:
            return True

        return False

    def loadModel(self):
        if self.topKGraphKey != None:
            self.model = torch.load(self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_model_top-'+str(self.topKGraphKey)+'.pt')
        else:
            self.model = torch.load(self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_model.pt')
        
        return self.model
    
    def saveModel(self):
        # create the folder on the given path if those don't exist
        if not os.path.exists(os.path.dirname(self.modelBasePath+self.subfolder)):
            os.makedirs(os.path.dirname(self.modelBasePath+self.subfolder))
        if self.topKGraphKey != None:
            torch.save(self.model, self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_model_top-'+str(self.topKGraphKey)+'.pt')
        else:
            torch.save(self.model, self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_model.pt')
    
    def checkIfOptimalParametersExist(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_optimal_params.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_optimal_params_top-'+str(self.topKGraphKey)+'.pt'
        
        if os.path.isfile(path) == True:
            return True
        else:
            return False
        
    def loadOptimalParameters(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_optimal_params.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_optimal_params_top-'+str(self.topKGraphKey)+'.pt'

        self.bestParams = torch.load(path)
        return self.bestParams
    
    def saveOptimalParameters(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_optimal_params.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_optimal_params_top-'+str(self.topKGraphKey)+'.pt'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.bestParams, path)
    
    def checkIfBestValidationAUCExists(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_best_val_auc.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_best_val_auc_top-'+str(self.topKGraphKey)+'.pt'

        if os.path.isfile(path=path) == True:
            return True
        else:
            return False
    
    def loadBestValidationAUC(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_best_val_auc.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_best_val_auc_top-'+str(self.topKGraphKey)+'.pt'
        data = torch.load(path)
        
        self.bestValAuc = data['bestValAuc']
        self.bestValAucList = data['bestValAucList']
        
        return self.bestValAuc
    
    def saveBestValidationAUC(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_best_val_auc.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_best_val_auc_top-'+str(self.topKGraphKey)+'.pt'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save({'bestValAuc': self.bestValAuc, 'bestValAucList': self.bestValAucList}, path)

    def getValidationResults(self):
        if self.checkIfModelExists():
            if not hasattr(self, 'model') or self.model == None:
                self.loadModel()
            val_loss = self.getValidationLoss()
            val_roc = self.test(dataList=self.testDataList, printStatistics=True)
            return val_loss, val_roc
        
    def trainModelAndGetValidationResults(self, continueTraining = False):
        if self.checkIfModelExists() & (continueTraining == False):
            pass
        elif self.checkIfModelExists() & (continueTraining == True):
            self.parametersGridSearch(continueTraining = True)
        else:
            self.parametersGridSearch()

        if hasattr(self, 'model') and self.model == None or not hasattr(self, 'model'):
            self.loadModel()
        return self.getValidationResults()

    def performTraining(self, params, gridSearchIteration = None, gridSearchTotalIterations = None, continueTraining = False):
        self.print(f"performing training with params: {params}")
        best_test_auc = 0
        if continueTraining == False or continueTraining == True and not self.checkIfModelExists():
            if self.topKGraphKey != None:
                self.model = MPXGAT_H(dataset=self.customGraph, heads=params['numHeads'], out_channels=params['outChannels'], hidden_channels=params['hiddenChannels']).to(self.device)
            else:
                self.model = MPXGAT_H(dataset=self.datasetUtils.getDataset(), heads=params['numHeads'], out_channels=params['outChannels'], hidden_channels=params['hiddenChannels']).to(self.device)
        else:
            if hasattr(self, 'model') and self.model == None or not hasattr(self, 'model'):
                self.model = self.loadModel()
            if self.checkIfBestValidationAUCExists():
                self.loadBestValidationAUC()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        loss_list_plot = []
        val_loss_list_plot = []
        val_auc_list_plot = []
        test_auc_list_plot = []
        
        num_epochs = params['numEpochs']

        pbar = tqdm(total=num_epochs, position=0, leave=True)
        early_stopper = EarlyStopper(patience=self.earlyStopingPatience, delta=0.6, delta_decay=0.01)
        for epoch in range(num_epochs):
            loss = self.train()

            test_auc, test_auc_list = self.test(self.testDataList)
            loss_validation = self.getValidationLoss()

            # Append the loss and AUC values to the lists
            loss_list_plot.append(loss.detach().cpu().numpy())
            val_loss_list_plot.append(loss_validation.detach().cpu().numpy())
            test_auc_list_plot.append(test_auc)

            if test_auc > best_test_auc:
                best_test_auc = test_auc
            # Update the progress bar and the description
            pbar.update(1)
            pbar.set_description(refresh=True, desc=f'Train Loss: {loss:.4f} Val Loss: {loss_validation:.4f}')
            pbar.set_postfix({'Test AUC': test_auc, 'GridSearchIteration': f'{gridSearchIteration}/{gridSearchTotalIterations}', 'ES delta': early_stopper.delta})
            if early_stopper.early_stop(loss_validation):
                self.print("Early stopping")
                break
            elif (epoch+1) % 20 == 0:
                early_stopper.decrease_delta() 
        
        pbar.close()
        self.print()
        self.print(f'{self.prefixModelName} final test auc: {test_auc}')
        self.print()
        self.print(f'{self.prefixModelName} final val auc: {test_auc}')
        if best_test_auc > self.bestValAuc:
            self.bestParams = params
            self.bestValAuc = best_test_auc
            self.bestValAucList = test_auc_list
            self.saveModel()
            # save the best params
            self.saveOptimalParameters()
            # save the best val auc
            self.saveBestValidationAUC()
            # Plot the loss and AUC values against the epoch number
            epoch_list = range(epoch+1)
            # create a plot with the loss and the AUC values
            fig = plt.figure()
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
            ax.plot(epoch_list, loss_list_plot, label='loss')
            ax.plot(epoch_list, test_auc_list_plot, label='test auc')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            # fig.show()
            if not os.path.exists(os.path.dirname(self.modelBasePath+self.subfolder)):
                os.makedirs(os.path.dirname(self.modelBasePath+self.subfolder))
            if self.topKGraphKey != None:
                fig.savefig(self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_model_'+str(self.topKGraphKey)+'.png')
            else:
                fig.savefig(self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_H_model.png')

            fig.clear()
            plt.close()
        
        return test_auc
            
    def getTrainEmbeddings(self):
        # get the final horizontal embeddings
        if not hasattr(self, 'model'):
            self.loadModel()

        with torch.no_grad():
            z_H = self.model.forward(horizontal_graphs=self.trainDataList)
        return z_H
    
    def getRandomEmbeddings(self):
        # get the final horizontal embeddings
        with torch.no_grad():
            z_H = self.model.forward(horizontal_graphs=self.trainDataList)
        
        z_H_random = []
        for i in range(len(z_H)):
            z_H_random.append(torch.rand(z_H[i].shape).to(self.device))

        return z_H_random
    
    def parametersGridSearch(self, continueTraining = False):
         # Perform grid search
        self.bestValAuc = 0
        if continueTraining == True and self.checkIfOptimalParametersExist():
            self.bestParams = self.loadOptimalParameters()
            # no need to consider again outChannels, hiddenChannels and numHeads since they are now part of the model that will be loaded
            self.parameterGrid['outChannels'] = [self.bestParams['outChannels']]
            self.parameterGrid['numHeads'] = [self.bestParams['numHeads']]
            self.parameterGrid['hiddenChannels'] = [self.bestParams['hiddenChannels']]
        grid = ParameterGrid(self.parameterGrid)
        gridSearchIteration = 0
        # get the total number of iterations for the grid search
        totalGridSearchIterations = len(grid)
        for params in grid:
            
            gridSearchIteration += 1
            
            self.performTraining(params=params, gridSearchIteration=gridSearchIteration, gridSearchTotalIterations=totalGridSearchIterations, continueTraining = continueTraining)
        
        # now that the grid seatch is finished, print the best parameters and the best validation auc
        self.print("Grid search results:")
        self.print("Best parameters:", self.bestParams)
        self.print(f'best horizontal val auc: {self.bestValAuc}')

        return self.bestValAuc