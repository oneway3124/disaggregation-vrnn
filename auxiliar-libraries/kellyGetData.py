from __future__ import print_function, division

# Stop matplotlib from drawing to X.
# Must be before importing matplotlib.pyplot or pylab!
import matplotlib
matplotlib.use('Agg')
import numpy as np
from lasagne.layers import (InputLayer, DenseLayer, ReshapeLayer,
                            DimshuffleLayer, Conv1DLayer)
from lasagne.nonlinearities import rectify

from neuralnilm.data.loadactivations import load_nilmtk_activations
from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
from neuralnilm.data.realaggregatesource import RealAggregateSource
from neuralnilm.data.datapipeline import DataPipeline
from neuralnilm.data.processing import DivideBy, IndependentlyCenter
from neuralnilm.net import Net
from neuralnilm.net import build_net
from neuralnilm.trainer import Trainer
from neuralnilm.metrics import Metrics
from neuralnilm.consts import DATA_FOLD_NAMES
from sklearn.model_selection import train_test_split

input_std = 0
target_std = 0
"""
Parameters: 
    num_batches_per_appliance: how many batches per appliance should be retrieved?
    numApp: Which appliance?
        -1: all
         0: Kettle
         1: Microwave
         2: Washing Machine
         3: Dish Washer
         4: Fridge
"""

def getNILMbatches(period, filename, target_inclusion_prob, windows, appliances, pTrain, pVal, pTest, num_seq_per_batch, seq_length, numApp):
    activations = load_nilmtk_activations(
        appliances=appliances,
        filename=filename,
        sample_period=period,
        windows=windows
    )

    filtered_activations = filter_activations(windows, appliances, activations)

    list_of_Xbatches = []
    list_of_Ybatches = []
    trainSize = int(num_seq_per_batch*pTrain)
    valSize = int(num_seq_per_batch*pVal)
    testSize = int(num_seq_per_batch*pTest)

    if(numApp == -1):
        print ("not implemented")
        #return None, None

        ##############3getbatch(enable_all_appliances=True)
        lenApps = len(appliances)

        trainSize = trainSize/lenApps
        valSize = valSizee/lenApps
        testSize = testSize/lenApps

        totalX = {'train':np.empty([0,self.time_steps]), 
                    'val':np.empty([0,self.time_steps]),
                    'test':np.empty([0,self.time_steps])}

        totalY = {'train':np.empty([0,self.time_steps,lenApps]), 
                    'val':np.empty([0,self.time_steps,lenApps]),
                    'test':np.empty([0,self.time_steps,lenApps])}

        for target_appliance in appliances:
            real_agg_source = RealAggregateSource(
                activations=filtered_activations,
                target_appliance=target_appliance,
                seq_length=seq_length,
                filename=filename,
                windows=windows,
                sample_period=period,
                target_inclusion_prob=target_inclusion_prob
            )
            #print('train')
            sampleTrain = real_agg_source.get_batch(num_seq_per_batch=trainSize, fold='train',validation=False).next()
            Xtrain = sampleTrain.before_processing
            input_std = Xtrain.input.flatten().std()
            target_std = Xtrain.target.flatten().std()
            input_processing=[DivideBy(input_std), IndependentlyCenter()]
            target_processing=[DivideBy(target_std)]
            Xtrain, Ytrain = Xtrain.input, Xtrain.target
            for step in input_processing:
                Xtrain = step(Xtrain)
            for step in target_processing:
                Ytrain = step(Xtrain)

            #print('validate')
            sampleVal = real_agg_source.get_batch(num_seq_per_batch=valSize, fold='val',validation=True).next()
            Xval = sampleVal.before_processing
            Xval, Yval = Xval.input,  Xval.target
            for step in input_processing:
                Xval = step(Xval)
            for step in target_processing:
                Yval = step(Yval)

            #print('test')
            sampleTest = real_agg_source.get_batch(num_seq_per_batch=testSize, fold='test',validation=True).next()
            Xtest = sampleTest.before_processing
            Xtest, Ytest = Xtest.input, Xtest.target
            for step in input_processing:
                Xtest = step(Xtest)
            for step in target_processing:
                Ytest = step(Ytest)

            '''
            pipeline = get_pipeline(period, filename, target_inclusion_prob, windows, appliances, appliances[numApp], activations, seq_length, num_seq_per_batch)
            batchTrain = pipeline.get_batch(fold='train',validation=False) #define sequence length in get_pipeline()
            batchVal = pipeline.get_batch(fold='val',validation=True)
            batchTest = pipeline.get_batch(fold='test',validation=True)
            '''

            '''
            print(Xtrain[0])
            print(Xtrain[499])
            print(Xval[0])
            print(Xval[249])
            print(Xtest[0])
            print(Xtest[249])
            '''
            totalX = {'train':np.squeeze(np.array(Xtrain)), 
                        'val':np.squeeze(np.array(Xval)),
                        'test':np.squeeze(np.array(Xtest))}

            totalY = {'train':np.squeeze(np.array(Ytrain)), 
                        'val':np.squeeze(np.array(Yval)),
                        'test':np.squeeze(np.array(Ytest))}
        
    else:
        target_appliance=appliances[numApp]
        real_agg_source = RealAggregateSource(
            activations=filtered_activations,
            target_appliance=target_appliance,
            seq_length=seq_length,
            filename=filename,
            windows=windows,
            sample_period=period,
            target_inclusion_prob=target_inclusion_prob
        )
        #print('train')
        sampleTrain = real_agg_source.get_batch(num_seq_per_batch=trainSize, fold='train',validation=False).next()
        Xtrain = sampleTrain.before_processing
        input_std = Xtrain.input.flatten().std()
        target_std = Xtrain.target.flatten().std()
        input_processing=[DivideBy(input_std), IndependentlyCenter()]
        target_processing=[DivideBy(target_std)]
        Xtrain, Ytrain = Xtrain.input, Xtrain.target
        for step in input_processing:
            Xtrain = step(Xtrain)
        for step in target_processing:
            Ytrain = step(Xtrain)

        #print('validate')
        sampleVal = real_agg_source.get_batch(num_seq_per_batch=valSize, fold='val',validation=True).next()
        Xval = sampleVal.before_processing
        Xval, Yval = Xval.input,  Xval.target
        for step in input_processing:
            Xval = step(Xval)
        for step in target_processing:
            Yval = step(Yval)

        #print('test')
        sampleTest = real_agg_source.get_batch(num_seq_per_batch=testSize, fold='test',validation=True).next()
        Xtest = sampleTest.before_processing
        Xtest, Ytest = Xtest.input, Xtest.target
        for step in input_processing:
            Xtest = step(Xtest)
        for step in target_processing:
            Ytest = step(Ytest)

        '''
        pipeline = get_pipeline(period, filename, target_inclusion_prob, windows, appliances, appliances[numApp], activations, seq_length, num_seq_per_batch)
        batchTrain = pipeline.get_batch(fold='train',validation=False) #define sequence length in get_pipeline()
        batchVal = pipeline.get_batch(fold='val',validation=True)
        batchTest = pipeline.get_batch(fold='test',validation=True)
        '''

        '''
        print(Xtrain[0])
        print(Xtrain[499])
        print(Xval[0])
        print(Xval[249])
        print(Xtest[0])
        print(Xtest[249])
        '''
        totalX = {'train':np.squeeze(np.array(Xtrain)), 
                    'val':np.squeeze(np.array(Xval)),
                    'test':np.squeeze(np.array(Xtest))}

        totalY = {'train':np.squeeze(np.array(Ytrain)), 
                    'val':np.squeeze(np.array(Yval)),
                    'test':np.squeeze(np.array(Ytest))}
        
    return totalX, totalY, input_std, target_std


def getBatches(period, filename, target_inclusion_prob, windows, appliances, num_batches, num_seq_per_batch, seq_length, numApp):
    activations = load_nilmtk_activations(
        appliances=appliances,
        filename=filename,
        sample_period=period,
        windows=windows
    )
    list_of_Xbatches = []
    list_of_Ybatches = []
    if(numApp == -1):
        for target_appliance in appliances:
            print("Getting batches for",target_appliance)
            pipeline = get_pipeline(period, filename, target_inclusion_prob, windows, appliances, target_appliance, activations, seq_length, num_seq_per_batch)
            for i in range(0,num_batches):
                batch = pipeline.get_batch() #define sequence length in get_pipeline()
                list_of_Xbatches.append(batch.input)
                list_of_Ybatches.append(batch.target)
    else:
        pipeline = get_pipeline(period, filename, target_inclusion_prob, windows, appliances, appliances[numApp], activations, seq_length, num_seq_per_batch)
        for i in range(0,num_batches):
            batch = pipeline.get_batch() #define sequence length in get_pipeline()
            list_of_Xbatches.append(batch.input)
            list_of_Ybatches.append(batch.target)
    return np.array(list_of_Xbatches),np.array(list_of_Ybatches)

# ------------------------ DATA ----------------------

def get_pipeline(period, data_path, target_inclusion_prob, windows, appliances, target_appliance, activations, seq_length, num_seq_per_batch):
    # Adding a and b to be coherent with buildings chosen in WINDOWS
    num_seq_per_batch = num_seq_per_batch
    filtered_activations = filter_activations(windows, appliances, activations)

    real_agg_source = RealAggregateSource(
        activations=filtered_activations,
        target_appliance=target_appliance,
        seq_length=seq_length,
        filename=data_path,
        windows=windows,
        sample_period=period,
        target_inclusion_prob=target_inclusion_prob
    )


    sample = real_agg_source.get_batch(num_seq_per_batch=seq_length).next()
    sample = sample.before_processing
    input_std = sample.input.flatten().std()
    target_std = sample.target.flatten().std()
    pipeline = DataPipeline(
        [real_agg_source], 
        num_seq_per_batch=num_seq_per_batch,
        input_processing=[DivideBy(input_std), IndependentlyCenter()],
        target_processing=[DivideBy(target_std)]
    )

    return pipeline


def select_windows(train_buildings, unseen_buildings):
    windows = {fold: {} for fold in DATA_FOLD_NAMES}

    def copy_window(fold, i):
        windows[fold][i] = WINDOWS[fold][i]

    for i in train_buildings:
        copy_window('train', i)
        copy_window('unseen_activations_of_seen_appliances', i)
    for i in unseen_buildings:
        copy_window('unseen_appliances', i)
    return windows


def filter_activations(windows, appliances, activations):
    new_activations = {
        fold: {appliance: {} for appliance in appliances}
        for fold in DATA_FOLD_NAMES}
    for fold, appliances in activations.iteritems():
        for appliance, buildings in appliances.iteritems():
            required_building_ids = windows[fold].keys()
            required_building_names = [
                'UK-DALE_building_{}'.format(i) for i in required_building_ids]
            for building_name in required_building_names:
                try:
                    new_activations[fold][appliance][building_name] = (
                        activations[fold][appliance][building_name])
                except KeyError:
                    pass
    return activations

def getKellyTrainValTest(period, filename, target_inclusion_prob, windows, appliances, numBatches, num_seq_per_batch, seq_length, numApp):

    X,Y = getBatches(period, filename, target_inclusion_prob, windows, appliances, numBatches, num_seq_per_batch, seq_length, numApp)

    #transform to 2D array
    X = X.reshape((X.shape[1],X.shape[2]))
    Y = Y.reshape((Y.shape[1],Y.shape[2]))

    #split data into train/validation/test (50%/25%/25%)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

    #33% of X_train_val is 25% of total
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.33, random_state=42)

    #return this as totalX, totalY, so load_csvdata returns the same thing

    totalX = {'train':X_train, 
                'val':X_val,
                'test':X_test}

    totalY = {'train':Y_train, 
                'val':Y_val,
                'test':Y_test}

    return totalX, totalY