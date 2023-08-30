import argparse
from mpxgat.utils.datasetUtils import DatasetUtils
from mpxgat.utils.horizontalModelUtils import HorizontalModelUtils
from mpxgat.utils.singleLayerUtils import SingleLayerUtils
from mpxgat.VGATModel import VGAT
from mpxgat.GATModel import GAT
import torch

NUMBER_OF_ITERATIONS = 2

MULTIPLEX_SAGE_DATASET_CUT = 1
RANDOM_HORIZONTAL_EMBEDDINGS = 2
RANDOM_SPLIT_DATASET_CUT = 3
TOP_K_LAYERS_DATASET = 4
RANDOM_MULTIPLEX_GRAPH = 5

DROSOPHILA = 0
ARXIV = 1
FFTWYT = 2
ALL_DATASETS = 3

RANDOM_DATASET = 3

datasetInfo = [
    {
        'datasetName': 'MultiplexGraphDrosophila',
        'datasetJsonName': 'multi-G_7_layers_1',
        'modelBasePath': 'models/drosophila/',
        'graphBasePath': 'Graphs/drosophila/',
        'minKLayers': 2,
        'maxKLayers': 7
    },
    {
        'datasetName': 'MultiplexGraphArxiv',
        'datasetJsonName': 'multi-G_13_layers_1',
        'modelBasePath': 'models/arxiv/',
        'graphBasePath': 'Graphs/arxiv/',
        'minKLayers': 2,
        'maxKLayers': 13
    },
    {
        'datasetName': 'MultiplexGraphFFTWYT',
        'datasetJsonName': 'multi-G_3_layers_1',
        'modelBasePath': 'models/fftwyt/',
        'graphBasePath': 'Graphs/fftwyt/',
        'minKLayers': 2,
        'maxKLayers': 3
    }
]

parser = argparse.ArgumentParser()
parser.add_argument('--print_dataset_info', type=bool, default=False, help='Prints the dataset info')
parser.add_argument('--experiment', type=int, default=MULTIPLEX_SAGE_DATASET_CUT, help='1: MultiplexSAGE dataset cut, 2: Random horizontal embeddings, 3: Random split dataset cut, 4: Top k layers dataset, 5: Increasing layer density')
parser.add_argument('--targetDataset', type=int, default=DROSOPHILA, help='0: Drosophila, 1: Arxiv, 2: FFTWYT, 3: All')
parser.add_argument('--number_of_iterations', type=int, default=NUMBER_OF_ITERATIONS, help='Number of iterations to run the experiment')
args = parser.parse_args()

printDatasetInfo = args.print_dataset_info
experiment_index = args.experiment
datasetTarget = args.targetDataset
numberOfIteration = args.number_of_iterations

def multiplexSAGEDatasetCut():
    print('----------------------- MULTIPLEX SAGE DATASET -----------------------')
    for iter in range(numberOfIteration):
        print(f'----------------------- ITERATION: {iter} -----------------------')
        for dataset in range(len(datasetInfo)):
            if datasetTarget != None and datasetTarget != ALL_DATASETS and datasetTarget != dataset:
                continue
            modelBasePath = datasetInfo[dataset]['modelBasePath']
            datasetName = datasetInfo[dataset]['datasetName']
            datasetJsonName = datasetInfo[dataset]['datasetJsonName']
            graphBasePath = datasetInfo[dataset]['graphBasePath']
            print(f'----------------------- DATASET: {datasetName} -----------------------')

            # ----------------------- DATASETS -----------------------
            datasetUtils: DatasetUtils = DatasetUtils(datasetName=datasetName, baseDataPath=graphBasePath, datasetJsonName=datasetJsonName, printDatasetInfo=printDatasetInfo, applyTransforms=False, augmentNodeFeatures=False)

            # ------------------- HORIZONTAL MODEL -------------------
            horizontalModelUtils = HorizontalModelUtils(modelBasePath=modelBasePath, modelName=datasetName, prefixModelName='00-HGAT', datasetUtils=datasetUtils, datasetKey=dataset, subfolder='multiplexSageCut/')
            horizontalModelUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
            
            horizontalEmbeddings = horizontalModelUtils.getTrainEmbeddings()
            del horizontalModelUtils
            torch.cuda.empty_cache()

            # ------------------- VERTICAL MODELS --------------------
            # ------------------- VGAT -------------------
            modelVGATUtils = SingleLayerUtils(datasetUtils=datasetUtils, modelName=datasetName, prefixModelName='01-VGAT-', modelBasePath=modelBasePath, modelClass=VGAT, horizontalEmbeddings=horizontalEmbeddings, datasetKey=dataset, subfolder='multiplexSageCut/')
            modelVGATUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
            
            del modelVGATUtils
            torch.cuda.empty_cache()
            
            # ------------------- GAT -------------------
            modelVGATUtils = SingleLayerUtils(datasetUtils=datasetUtils, modelName=datasetName, prefixModelName='02-GAT-', modelBasePath=modelBasePath, modelClass=GAT, horizontalEmbeddings=horizontalEmbeddings, datasetKey=dataset, subfolder='multiplexSageCut/')
            modelVGATUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
            
            del modelVGATUtils
            torch.cuda.empty_cache()
            print(f'----------------------- END DATASET: {datasetName} -----------------------')
        print(f'----------------------- END ITERATION: {iter} -----------------------')
        
def randomHorizontalEmbeddings():
    print('----------------------- RANDOM HORIZONTAL EMBEDDINGS -----------------------')
    for iter in range(numberOfIteration):
        print(f'----------------------- ITERATION: {iter} -----------------------')
        for dataset in range(len(datasetInfo)):
            if datasetTarget != None and datasetTarget != ALL_DATASETS and datasetTarget != dataset:
                continue
            modelBasePath = datasetInfo[dataset]['modelBasePath']
            datasetName = datasetInfo[dataset]['datasetName']
            datasetJsonName = datasetInfo[dataset]['datasetJsonName']
            graphBasePath = datasetInfo[dataset]['graphBasePath']
            print(f'----------------------- DATASET: {datasetName} -----------------------')

            # ----------------------- DATASETS -----------------------
            datasetUtils: DatasetUtils = DatasetUtils(datasetName=datasetName, baseDataPath=graphBasePath, datasetJsonName=datasetJsonName, printDatasetInfo=printDatasetInfo, applyTransforms=True, augmentNodeFeatures=True)

            # ------------------- HORIZONTAL MODEL -------------------
            horizontalModelUtils = HorizontalModelUtils(modelBasePath=modelBasePath, modelName=datasetName, prefixModelName='00-HGAT', datasetUtils=datasetUtils, datasetKey=dataset, subfolder='randomHEmbeddings/', datasetCut=False)
            horizontalModelUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
            horizontalEmbeddings = horizontalModelUtils.getRandomEmbeddings()
            del horizontalModelUtils
            torch.cuda.empty_cache()

            # ------------------- VERTICAL MODELS --------------------
            # ------------------- VGAT -------------------
            modelVGATUtils = SingleLayerUtils(datasetUtils=datasetUtils, modelName=datasetName, prefixModelName='01-VGAT-', modelBasePath=modelBasePath, modelClass=VGAT, horizontalEmbeddings=horizontalEmbeddings, datasetKey=dataset, subfolder='randomHEmbeddings/', datasetCut=False)
            modelVGATUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
            
            del modelVGATUtils
            torch.cuda.empty_cache()
            
            # ------------------- GAT -------------------
            modelVGATUtils = SingleLayerUtils(datasetUtils=datasetUtils, modelName=datasetName, prefixModelName='02-GAT-', modelBasePath=modelBasePath, modelClass=GAT, horizontalEmbeddings=horizontalEmbeddings, datasetKey=dataset, subfolder='randomHEmbeddings/', datasetCut=False)
            modelVGATUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
            
            del modelVGATUtils
            torch.cuda.empty_cache()
            print(f'----------------------- END DATASET: {datasetName} -----------------------')
        print(f'----------------------- END ITERATION: {iter} -----------------------')
        
def randomSplitDatasetCut():
    print('----------------------- RANDOM SPLIT DATASET CUT -----------------------')
    for iter in range(numberOfIteration):
        print(f'----------------------- ITERATION: {iter} -----------------------')
        for dataset in range(len(datasetInfo)):
            if datasetTarget != None and datasetTarget != ALL_DATASETS and datasetTarget != dataset:
                continue
            modelBasePath = datasetInfo[dataset]['modelBasePath']
            datasetName = datasetInfo[dataset]['datasetName']
            datasetJsonName = datasetInfo[dataset]['datasetJsonName']
            graphBasePath = datasetInfo[dataset]['graphBasePath']
            
            subfolder = 'randomSplitDatasetCut/'
            print(f'----------------------- DATASET: {datasetName} -----------------------')

            # ----------------------- DATASETS -----------------------
            datasetUtils: DatasetUtils = DatasetUtils(datasetName=datasetName, baseDataPath=graphBasePath, datasetJsonName=datasetJsonName, printDatasetInfo=printDatasetInfo, applyTransforms=True, augmentNodeFeatures=True)

            # ------------------- HORIZONTAL MODEL -------------------
            horizontalModelUtils = HorizontalModelUtils(modelBasePath=modelBasePath, modelName=datasetName, prefixModelName='00-HGAT', datasetUtils=datasetUtils, datasetKey=dataset, subfolder=subfolder, datasetCut=False)
            horizontalModelUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
            
            horizontalEmbeddings = horizontalModelUtils.getTrainEmbeddings()
            del horizontalModelUtils
            torch.cuda.empty_cache()

            # ------------------- VERTICAL MODELS --------------------
            # ------------------- VGAT -------------------
            modelVGATUtils = SingleLayerUtils(datasetUtils=datasetUtils, modelName=datasetName, prefixModelName='01-VGAT-', modelBasePath=modelBasePath, modelClass=VGAT, horizontalEmbeddings=horizontalEmbeddings, datasetKey=dataset, subfolder=subfolder, datasetCut=False)
            modelVGATUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
            
            del modelVGATUtils
            torch.cuda.empty_cache()
            
            # ------------------- GAT -------------------
            modelVGATUtils = SingleLayerUtils(datasetUtils=datasetUtils, modelName=datasetName, prefixModelName='02-GAT-', modelBasePath=modelBasePath, modelClass=GAT, horizontalEmbeddings=horizontalEmbeddings, datasetKey=dataset, subfolder=subfolder, datasetCut=False)
            modelVGATUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
            
            del modelVGATUtils
            torch.cuda.empty_cache()
            print(f'----------------------- END DATASET: {datasetName} -----------------------')
        print(f'----------------------- END ITERATION: {iter} -----------------------')
        
def topKLayersDataset():
    customOrderList = [ # (original layer, new layer)
        # DROSOPHILA
        [(6,1),(3,0),(2,4),(4,3),(5,2),(1,6),(0,5)],
        # # ARXIV
        [(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(11,11),(12,12)]
    ]
    print('----------------------- TOP K LAYERS DATASET -----------------------')
    for iter in range(numberOfIteration):
        print(f'----------------------- ITERATION: {iter} -----------------------')
        for dataset in range(len(datasetInfo)):
            if datasetTarget != None and datasetTarget != ALL_DATASETS and datasetTarget != dataset:
                continue
            modelBasePath = datasetInfo[dataset]['modelBasePath']
            datasetName = datasetInfo[dataset]['datasetName']
            datasetJsonName = datasetInfo[dataset]['datasetJsonName']
            graphBasePath = datasetInfo[dataset]['graphBasePath']
            minKLayers = datasetInfo[dataset]['minKLayers']
            maxKLayers = datasetInfo[dataset]['maxKLayers']
            datasetUtils: DatasetUtils = DatasetUtils(datasetName=datasetName, baseDataPath=graphBasePath, datasetJsonName=datasetJsonName, printDatasetInfo=printDatasetInfo, applyTransforms=False, augmentNodeFeatures=False)
            datasetUtils.setApplyTransforms(True)
            datasetUtils.augmentNodeFeatures = True
            for toplayers in range (minKLayers,maxKLayers+1):
                print(f'----------------------- DATASET: {datasetName} TOP: {toplayers} -----------------------')
                topKGraph = datasetUtils.getTopKLayers(k=toplayers , customOrder=customOrderList[dataset]) # 
                # ------------------- HORIZONTAL MODEL -------------------
                horizontalModelUtils = HorizontalModelUtils(modelBasePath=modelBasePath, modelName=datasetName, prefixModelName='00-HGAT', datasetUtils=datasetUtils, datasetKey=dataset, customGraph=topKGraph, topKGraphKey=toplayers, subfolder='topKLayers_test5/', datasetCut=False)
                horizontalModelUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
                horizontalEmbeddings = horizontalModelUtils.getTrainEmbeddings()
                del horizontalModelUtils
                torch.cuda.empty_cache()

                # ------------------- VERTICAL MODELS --------------------
                # ------------------- VGAT -------------------
                modelVGATUtils = SingleLayerUtils(datasetUtils=datasetUtils, modelName=datasetName, prefixModelName='01-VGAT-', modelBasePath=modelBasePath, modelClass=VGAT, horizontalEmbeddings=horizontalEmbeddings, datasetKey=dataset, customGraph=topKGraph, topKGraphKey=toplayers, subfolder='topKLayers_test5/', datasetCut=False)
                modelVGATUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
                del modelVGATUtils
                torch.cuda.empty_cache()
                # ------------------- GAT -------------------
                modelVGATUtils = SingleLayerUtils(datasetUtils=datasetUtils, modelName=datasetName, prefixModelName='02-GAT-', modelBasePath=modelBasePath, modelClass=GAT, horizontalEmbeddings=horizontalEmbeddings, datasetKey=dataset, customGraph=topKGraph, topKGraphKey=toplayers, subfolder='topKLayers_test5/', datasetCut=False)
                modelVGATUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True)
                del modelVGATUtils
                torch.cuda.empty_cache()

def randomMultiplexGraph():
    p_h_list = [
        [0.02, 0.02, 0.02, 0.02, 0.02],     # baseline
    ]
    
    p_v_list = [
        0.02    #baseline
    ]
    
    numLayers = 6
    
    modelBasePath = 'models/random_test_deep_model_no_bias/'
    datasetName = 'random'
    graphBasePath = 'Graphs/random_test_deep_model_no_bias/'
    
    
    print('----------------------- RANDOM GRAPH -----------------------')
    for iter in range(numberOfIteration):
        print(f'----------------------- ITERATION: {iter} -----------------------')
        experiment_index = 0
        # ----------------------- DATASETS -----------------------
        datasetUtils: DatasetUtils = DatasetUtils(datasetName=datasetName, printDatasetInfo=printDatasetInfo, applyTransforms=True, augmentNodeFeatures=True, baseDataPath=graphBasePath, useRandomGraph=True)
        customGraph = datasetUtils.generateRandomMultiplexGraph(p_h=p_h_list[experiment_index], p_v=p_v_list[experiment_index], numLayers=len(p_h_list[experiment_index]), numNodes=100)
        
        
        # ------------------- HORIZONTAL MODEL -------------------
        horizontalModelUtils = HorizontalModelUtils(modelBasePath=modelBasePath, modelName=datasetName, prefixModelName='00-HGAT', datasetUtils=datasetUtils, datasetKey=RANDOM_DATASET, subfolder=f'randomGraph50LayerModel/', datasetCut=False, customGraph=customGraph)
        horizontalModelUtils.trainModelAndGetValidationResults(continueTraining=False ) # if iter>0 else True
        
        horizontalEmbeddings = horizontalModelUtils.getTrainEmbeddings()
        del horizontalModelUtils
        torch.cuda.empty_cache()

        # ------------------- VERTICAL MODELS --------------------
        # ------------------- VGAT -------------------
        modelVGATUtils = SingleLayerUtils(datasetUtils=datasetUtils, modelName=datasetName, prefixModelName='01-VGAT-', modelBasePath=modelBasePath, modelClass=VGAT, horizontalEmbeddings=horizontalEmbeddings, datasetKey=RANDOM_DATASET, subfolder=f'randomGraph50LayerModel/', datasetCut=False, customGraph=customGraph)
        modelVGATUtils.trainModelAndGetValidationResults(continueTraining=False if iter>0 else True) # 
        
        del modelVGATUtils
        torch.cuda.empty_cache()
        
        print(f'----------------------- END ITERATION: {iter} -----------------------')
        

switch_experiment = {
    MULTIPLEX_SAGE_DATASET_CUT: multiplexSAGEDatasetCut,
    RANDOM_HORIZONTAL_EMBEDDINGS: randomHorizontalEmbeddings,
    RANDOM_SPLIT_DATASET_CUT: randomSplitDatasetCut,
    TOP_K_LAYERS_DATASET: topKLayersDataset,
    RANDOM_MULTIPLEX_GRAPH: randomMultiplexGraph
}

experiment = switch_experiment.get(experiment_index, lambda: "Invalid experiment")
experiment()