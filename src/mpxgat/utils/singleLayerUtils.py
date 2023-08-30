import datetime
import json
import pandas as pd
import seaborn as sns
from mpxgat.utils.datasetUtils import DatasetUtils
from torch_geometric.utils import negative_sampling
import torch
import torch.nn.functional as F
import os.path
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.EarlyStopper import EarlyStopper
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score,  roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import ParameterGrid
from mpxgat.multiplex_graph import MultiplexGraph
import copy
import numpy as np

class SingleLayerUtils:
    
    datasetCut: bool
    """ flag to indicate if the train/test split to be used is the one provided by the dataset or a random one """
    
    earlyStopingPatience: int = 5
    """ number of epochs to wait before stopping the training if the validation loss does not improve """
    
    customGraph: MultiplexGraph = None
    """ multiplex graph ordered by the number of nodes in each layer """
    
    topKGraphKey: int = None
    """ number of ordered layers to be used in the topKGraph (customGraph)"""
    
    subfolder: str = None
    """ subfolder where to save the results """
    
    parameterGridList = [
        { #drosophila
            'hiddenChannels': [128], 'lr': [0.001], 'numEpochs': [1500], 'numHeads': [4], 'outChannels': [128], 'weight_decay': [0.0005]
        },
        { #arxiv
            'hiddenChannels': [256], 'lr': [0.0005], 'numEpochs': [1500], 'numHeads':[3], 'outChannels': [128], 'weight_decay': [5e-05]
        },
        { #fftwyt
            'hiddenChannels': [256], 'lr': [0.0005], 'numEpochs': [1500], 'numHeads': [3], 'outChannels': [128], 'weight_decay':[ 5e-05]
        },
        { #random
            'hiddenChannels': [256], 'lr': [0.001], 'numEpochs': [1500], 'numHeads': [4], 'outChannels': [128], 'weight_decay': [5e-05]
        }
    ]
    
    def print(self, *args):
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " " + self.prefixModelName + self.modelName + " :: ", *args)

    def __init__(self, modelBasePath = None, modelName = None, prefixModelName='', model = None, datasetUtils: DatasetUtils = None, horizontalEmbeddings = None, modelClass = None, datasetKey = None, customGraph : MultiplexGraph = None, topKGraphKey = None, subfolder = '', datasetCut = True):
        self.datasetKey = datasetKey
        self.parameterGrid = self.parameterGridList[self.datasetKey]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.datasetUtils: DatasetUtils = datasetUtils
        self.datasetCut = datasetCut
        
        self.horizontalEmbeddings = horizontalEmbeddings
        self.z_H = self.horizontalEmbeddings
        # normalize each row of the horizontal embeddings
        self.z_H = torch.cat(self.z_H, dim=0)
        self.z_H = F.normalize(self.z_H, p=2, dim=1)

        self.modelBasePath = modelBasePath
        self.modelName = modelName
        self.modelClass = modelClass
        self.prefixModelName = prefixModelName
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        self.customGraph = customGraph
        self.topKGraphKey = topKGraphKey
        self.subfolder = subfolder
        
        self.trainData = None
        self.testData = None
        if self.topKGraphKey != None:
            self.data = self.customGraph.get_vertical_graph()
            self.trainData = self.data[0]
            self.testData = self.data[1]
        else:
            if self.datasetCut == True:
                self.data = datasetUtils.getVerticalData()
            else :
                self.trainData, self.testData = self.datasetUtils.getVerticalData()
                
        self.initTrainTestData()

        # print informations about the train and test datasets
        self.print_dataset_info(self.trainData, 'train')
        self.print_dataset_info(self.testData, 'test')
        
    def initTrainTestData(self):
        if self.datasetCut == True:
            self.trainData = copy.deepcopy(self.data)
            self.trainData.edge_index = self.trainData.edge_index[:, self.data.train_mask]
            self.trainData.edge_label_index = copy.deepcopy(self.trainData.edge_index)
            self.trainData.edge_label = torch.ones(self.trainData.edge_label_index.shape[1])
            neg_edge_index = negative_sampling(edge_index=self.trainData.edge_index, num_nodes=self.trainData.num_nodes, num_neg_samples=self.trainData.num_edges)
            self.trainData.edge_label_index = torch.cat([self.trainData.edge_label_index, neg_edge_index], dim=-1).to(self.device)
            self.trainData.edge_label = torch.cat([self.trainData.edge_label, torch.zeros(neg_edge_index.shape[1])], dim=-1).to(self.device)

            self.testData = copy.deepcopy(self.data)
            self.testData.edge_index = self.testData.edge_index[:, self.data.test_mask]
            self.testData.edge_label_index = copy.deepcopy(self.testData.edge_index)
            self.testData.edge_label = torch.ones(self.testData.edge_label_index.shape[1])
            neg_edge_index = negative_sampling(edge_index=self.testData.edge_index, num_nodes=self.testData.num_nodes, num_neg_samples=self.testData.num_edges)
            self.testData.edge_label_index = torch.cat([self.testData.edge_label_index, neg_edge_index], dim=-1).to(self.device)
            self.testData.edge_label = torch.cat([self.testData.edge_label, torch.zeros(neg_edge_index.shape[1])], dim=-1).to(self.device)
        else:
            return

    def print_dataset_info(self, dataset, prefix):
        print('\n\n\n====================')
        print(f'{prefix} Number of nodes: {dataset.x.shape[0]}')
        try:
            print(f'{prefix} Number of node features: {dataset.x.shape[1]}')
        except:
            print(f'{prefix} no node features')
        print(f'{prefix} Number of edges: {dataset.edge_index.shape[1]}')
        if (dataset.edge_attr != None):
            print(f'{prefix} Number of edge features: {dataset.edge_attr.shape[1]}')
        print(f'{prefix} Average node degree: {dataset.edge_index.shape[1] / dataset.x.shape[0]:.2f}')
        print(f'{prefix} Has isolated nodes: {dataset.has_isolated_nodes()}')
        print(f'{prefix} Has self-loops: {dataset.has_self_loops()}')
        print(f'{prefix} Is undirected: {dataset.is_undirected()}')
        print("#####\n")
        print('====================\n\n\n')
    
    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        # compute the forward pass of the vertical graph, this will output the node embeddings for this train epoch.
        z_V = self.model.forward(horizontal_layer_embeddings=self.z_H, vertical_graph=self.trainData)
        # performing a negative sampling
        train_data = self.trainData
        if self.datasetCut == True:
            # calculate the predicted output of the model as dot product between the embeddings of the two nodes of the edge
            out = self.model.decode(z_V, train_data.edge_label_index).view(-1)
            # calculate the loss for the vertical graph
            loss = self.criterion(out, train_data.edge_label)
        else:
            neg_edge_index = negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes, num_neg_samples=train_data.num_edges)
            # concatenate the negative sampled edges to the positive
            edge_label_index = torch.cat(
                [train_data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            # concatenate the negative sampled labels to the positive ones
            edge_label = torch.cat([
                train_data.edge_label,
                train_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)
            # calculate the predicted output of the model as dot product between the embeddings of the two nodes of the edge
            out = self.model.decode(z_V, edge_label_index).view(-1)
            # calculate the loss for the vertical graph
            loss = self.criterion(out, edge_label)

        # propagate the loss for the vertical graph
        loss.backward()    
        # update the weights of the model
        self.optimizer.step()
        return loss
    
    @torch.no_grad()
    def getValidationLoss(self):
        test_data = self.testData
        # compute the forward pass of the vertical graph, this will output the node embeddings for this train epoch.
        z_V = self.model.forward(horizontal_layer_embeddings=self.z_H, vertical_graph=test_data)
        # performing a negative sampling
        if self.datasetCut == True:
            # calculate the predicted output of the model as dot product between the embeddings of the two nodes of the edge
            out = self.model.decode(z_V, test_data.edge_label_index).view(-1)
            loss = self.criterion(out, test_data.edge_label)
        else:
            neg_edge_index = negative_sampling(edge_index=test_data.edge_index, num_nodes=test_data.num_nodes, num_neg_samples=test_data.num_edges)
            # concatenate the negative sampled edges to the positive
            edge_label_index = torch.cat(
                [test_data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            # concatenate the negative sampled labels to the positive ones
            edge_label = torch.cat([
                test_data.edge_label,
                test_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)
            # calculate the predicted output of the model as dot product between the embeddings of the two nodes of the edge
            out = self.model.decode(z_V, edge_label_index).view(-1)
            # calculate the loss for the vertical graph
            loss = self.criterion(out, edge_label)
        return loss

    @torch.no_grad()
    def test(self, model, data, printStatistics = False):
        z_V = model.forward(horizontal_layer_embeddings=self.z_H, vertical_graph=data)

        # the sigmoid function is applied to the output of the model in order to obtain a value between 0 and 1 (probability of the edge to be positive)
        out = model.decode(z_V, data.edge_label_index).view(-1).sigmoid()
        roc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
        
        if printStatistics == True:
            y_true = data.edge_label.cpu().numpy()
            y_pred_prob = out.cpu().numpy()
            roc_auc = roc_auc_score(y_true, y_pred_prob)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
            # get the optimal threshold (the one that minimizes the distance between the ROC curve and the top-left corner of the plot)
            optimal_idx = np.argmax(tpr - fpr) 
            threshold = thresholds[optimal_idx]
            # threshold = 0.6
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
            self.print("ROC AUC score:", roc)
            self.print("Precision:", precision)
            self.print("Recall:", recall)
            self.print("F1 score:", f1)
            self.print("Accuracy:", accuracy)
            # ------------------------------ CONFUSION MATRIX ------------------------------
            # Create heatmap of confusion matrix
            fig, ax = plt.subplots(1, 1, figsize=(9, 10))
            ConfusionMatrixDisplay(cm).plot(cmap='Greens', ax=ax)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')
            ax.set_title("Confusion matrix")

            # Save and close the figure
            outpath = self.modelBasePath +self.subfolder + self.prefixModelName + self.modelName + "_confusion_matrix.png"
            if self.topKGraphKey != None:
                outpath = self.modelBasePath + self.subfolder + self.prefixModelName + self.modelName + "_" + str(self.topKGraphKey) + "_confusion_matrix.png"
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.outpath.dirname(outpath))
            plt.savefig(outpath)
            plt.clf()
            plt.close()
            
            # ------------------------------- CLASSIFICATION REPORT ------------------------------------
            # plot the ROC curve
            fig = plt.figure(2, figsize=(9, 6))
            ax = fig.add_subplot(111)
            labels = [0, 1]
            target_names = ['Absent', 'Present']
            y_pred = out_binary.cpu().numpy()
            class_report = classification_report(y_true, y_pred, target_names=target_names, labels=labels, output_dict=True)
            # .iloc[:-1, :] to exclude support
            sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True)
            # save the plot
            outpath = self.modelBasePath + self.subfolder + self.prefixModelName + self.modelName + "_ClassificationReport.png"
            if self.topKGraphKey != None:
                outpath = self.modelBasePath + self.subfolder + self.prefixModelName + self.modelName + "_" + str(self.topKGraphKey) + "_ClassificationReport.png"
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.outpath.dirname(outpath))
            plt.savefig(outpath)
            plt.clf()
            plt.close()
            self.print("Classification Report:\n", class_report)
            
            # ------------------------------- ROC CURVE ------------------------------------
            fig = plt.figure(3, figsize=(9, 6))
            ax = fig.add_subplot(111)
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            outpath = self.modelBasePath + self.subfolder + self.prefixModelName + self.modelName + "_roc.png"
            if self.topKGraphKey != None:
                outpath = self.modelBasePath + self.subfolder + self.prefixModelName + self.modelName + "_" + str(self.topKGraphKey) + "_roc.png"
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.outpath.dirname(outpath))

            plt.savefig(outpath)
            plt.clf()
            plt.close()
            
            # create a dictionary with the collected statistics and save it to a json file
            statistics = {
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "f1": f1,
                "roc": roc,
                "confusion_matrix": cm.tolist(),
                "threshold": str(threshold),
                "classification_report": class_report,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # add timestamp to the statistics
            }
            path = self.modelBasePath + self.subfolder + self.prefixModelName + self.modelName + "_V_statistics.json"
            if self.topKGraphKey != None:
                path = self.modelBasePath + self.subfolder + self.prefixModelName + self.modelName + "_" + str(self.topKGraphKey) + "_V_statistics.json"
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            with open(path, "a") as outfile:
                json.dump(statistics, outfile)
                outfile.write("\n")
                
            path = self.modelBasePath + self.subfolder + self.prefixModelName + self.modelName + "_V_ROC.txt"
            if self.topKGraphKey != None:
                path = self.modelBasePath + self.subfolder + self.prefixModelName + self.modelName + "_" + str(self.topKGraphKey) + "_V_ROC.txt"
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            with open(path, "a") as outfile:
                json.dump(roc, outfile)
                outfile.write("\n")

        return roc
    
    def checkIfModelExists(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_model.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_'+str(self.topKGraphKey)+'_V_model.pt'
        if os.path.isfile(path) == True:
            return True
        else:
            return False

    def loadModel(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_model.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_'+str(self.topKGraphKey)+'_V_model.pt'
        self.model = torch.load(path)
        return self.model
    
    def saveModel(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_model.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_'+str(self.topKGraphKey)+'_V_model.pt'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.model, path)
    
    def checkIfOptimalParametersExist(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_optimal_params.pt'
        if os.path.isfile(path) == True:
            return True
        else:
            return False
        
    def loadOptimalParameters(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_optimal_params.pt'
        self.bestParams = torch.load(path)
        return self.bestParams
    
    def saveOptimalParameters(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_optimal_params.pt'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.bestParams, path)
    
    def checkIfBestValidationAUCExists(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_best_val_auc.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_'+str(self.topKGraphKey)+'_V_best_val_auc.pt'
        if os.path.isfile(path) == True:
            return True
        else:
            return False
    
    def loadBestValidationAUC(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_best_val_auc.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_'+str(self.topKGraphKey)+'_V_best_val_auc.pt'
        self.bestValAuc = torch.load(path)
        return self.bestValAuc
    
    def saveBestValidationAUC(self):
        path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_best_val_auc.pt'
        if self.topKGraphKey != None:
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_'+str(self.topKGraphKey)+'_V_best_val_auc.pt'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.bestValAuc, path)

    def getValidationResults(self):
        if self.checkIfModelExists():
            if not hasattr(self, 'model') or self.model == None:
                self.loadModel()
            val_loss = self.getValidationLoss()
            val_roc = self.test(self.model, self.testData, True)
            return val_loss, val_roc
        
    def trainModelAndGetValidationResults(self, continueTraining = False):
        if self.checkIfModelExists() & (continueTraining == False):
            pass
        elif self.checkIfModelExists() & (continueTraining == True):
            self.parametersGridSearch(continueTraining = True)
        else:
            self.parametersGridSearch()
        if not hasattr(self, 'model') or self.model == None:
            self.loadModel()
        return self.getValidationResults()

    def performTraining(self, params, gridSearchIteration = None, gridSearchTotalIterations = None, continueTraining = False):
        self.print(f"performing training with params: {params}")
        self.z_H = self.horizontalEmbeddings
        best_test_auc = 0
        # normalize each row of the horizontal embeddings
        self.z_H = torch.cat(self.z_H, dim=0)
        self.z_H = F.normalize(self.z_H, p=2, dim=1)

        if continueTraining == False or continueTraining == True and not self.checkIfModelExists():
            self.model = self.modelClass(heads=params['numHeads'], out_channels=params['outChannels'], hidden_channels=params['hiddenChannels']).to(self.device)
        else:
            if not hasattr(self, 'model') or self.model == None:
                self.model = self.loadModel()
            if self.checkIfBestValidationAUCExists():
                self.loadBestValidationAUC()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        loss_list_plot = []
        test_auc_list_plot = []
        
        num_epochs = params['numEpochs']

        pbar = tqdm(total=num_epochs, position=0, leave=True)
        early_stopper = EarlyStopper(patience=5, delta=0.3, delta_decay=0.01)
        for epoch in range(num_epochs):
            loss = self.train()
            test_auc = self.test(self.model, self.testData)
            loss_validation = self.getValidationLoss()

            # Append the loss and AUC values to the lists
            loss_list_plot.append(loss.detach().cpu().numpy())
            test_auc_list_plot.append(test_auc)
            
            if test_auc > best_test_auc:
                best_test_auc = test_auc
            # Update the progress bar and the description
            pbar.update(1)
            pbar.set_description(refresh=True, desc=f'Train Loss: {loss:.4f} Val Loss: {loss_validation:.4f}, GridSearchIteration: {gridSearchIteration}/{gridSearchTotalIterations}')
            pbar.set_postfix({'Test AUC': test_auc,'ES delta': early_stopper.delta})
            if early_stopper.early_stop(loss_validation):
                self.print("Early stopping")
                break
            elif (epoch+1) % 20 == 0:
                early_stopper.decrease_delta() 
            
        pbar.close()
        self.print()
        self.print(f'{self.prefixModelName} final vertical test auc: {test_auc}')
        self.print()
        if best_test_auc > self.bestValAuc:
            self.bestParams = params
            self.bestValAuc = best_test_auc
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
            path = self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_V_model.png'
            if self.topKGraphKey != None:
                path= self.modelBasePath+self.subfolder+self.prefixModelName+self.modelName+'_'+str(self.topKGraphKey)+'_V_model.png'
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            fig.savefig(path)
            fig.clear()
            plt.close()
        
        return test_auc
 
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
        self.print(f'best vertical val auc: {self.bestValAuc}')

        return self.bestValAuc