from __future__ import division
import os
import numpy as np
import cPickle
import fnmatch
import re
import dataSet_ts as dt
import datetime
import pickle
from lxml import etree


def fetch_ukdale(data_path, windows, appliances, numApps, period, n_steps, stride_train, stride_test, typeLoad=0, seq_per_batch=0, flgAggSumScaled=0, 
                  flgFilterZeros=0, target_inclusion_prob=0.5, trainPer=0.5, valPer=0.25, testPer=0.25):

     
    '''
    Deleting huge part of seems like generating data from other metadata
    '''
    reader = dt.ReaderTS(windows, appliances, n_steps, stride_train, stride_test, period, flgAggSumScaled, flgFilterZeros,
                        flgScaling=1, trainPer=trainPer, valPer=valPer, testPer=testPer)
    print(windows)
    if (numApps==-1):
        truFileName='all_'+str(n_steps)+'_tr'+str(trainPer)+'_te'+str(testPer)+'_val'+str(valPer)+'_b'+str(len(windows))+'_'+str(windows[1][0])+'_'+str(windows[1][1])
    else:
      if (typeLoad==1):
        truFileName=appliances[numApps]+'_'+str(n_steps)+'_tr'+str(windows['train'][1][0])+'_te'+str(windows['test'][1][0])+'_val'+str(windows['val'][1][0])+'_'+str(windows['val'][1][1])
      else:
        truFileName=appliances[numApps]+'_'+str(n_steps)+'_'+str(windows[1][0])+'_'+str(windows[1][1])
    try:
        split = pickle.load( open(data_path+"/pickles/"+truFileName+".pickle","rb"))
        return split['X_train'], split['Y_train'], split['X_val'], split['Y_val'], split['X_test'], split['Y_test'], reader
    except (OSError, IOError) as e:
        XdataSet, YdataSet = reader.load_csvdata(data_path, numApps, typeLoad, seq_per_batch, target_inclusion_prob)
    #shape before: batch, apps, steps
        x_train, x_val, x_test, y_train, y_val, y_test = XdataSet['train'],XdataSet['val'],XdataSet['test'],YdataSet['train'],YdataSet['val'],YdataSet['test']
        x_train, x_val, x_test = np.expand_dims(x_train,axis=2), np.expand_dims(x_val,axis=2), np.expand_dims(x_test,axis=2)
        if (numApps!=-1):
          y_train,y_val,y_test = np.expand_dims(y_train,axis=2), np.expand_dims(y_val,axis=2), np.expand_dims(y_test,axis=2)
        with open(data_path+"/pickles/"+truFileName+".pickle",'wb') as splitWrite:
            pickle.dump({'X_train':x_train,'Y_train':y_train,'X_val':x_val,'Y_val':y_val,'X_test':x_test,'Y_test':y_test},splitWrite)

    return x_train, y_train, x_val, y_val, x_test,y_test, reader

def build_dict_instances_plot(listDates, windows, reader, sizeBatch, TestSize):
  maxBatch = TestSize/sizeBatch  - 1
  listInst = []
  for strDate in listDates:
    initialDate = datetime.datetime.strptime(windows[1][0], '%Y-%m-%d')
    targetDate = datetime.datetime.strptime(strDate, '%Y-%m-%d %H:%M')
    nInstance = (targetDate - initialDate).total_seconds()/6
    listInst.append(nInstance)

  instancesPlot = {}
  tupBinsOffset = []
  print(listInst)
  for inst in listInst:
    if (reader.bin_edges[2]< inst and inst<reader.bin_edges[3]): # If the instance is in the test set
      offSet = int((inst - reader.bin_edges[2])/reader.time_steps)
      tupBinsOffset.append((2,offSet))
  print(tupBinsOffset)
  dictInstances = {}
  for tupIns in tupBinsOffset:
    smallerIndexes = np.array(reader.idxFiltered[tupIns[0]]) #Convert the indexs (tuples) in an array
    tupNumberSmaller = np.where(smallerIndexes<tupIns[1]) #Find the number of indexes that are smaller than the offset
    indexAfterFilter = len(tupNumberSmaller[0])
    nBatch = int(indexAfterFilter/sizeBatch) - 1
    indexAfterBatch =  (indexAfterFilter % sizeBatch) -1
    if (nBatch <= maxBatch):
      if nBatch in dictInstances:
        dictInstances[nBatch].append(int(indexAfterBatch))
      else:
        dictInstances[nBatch] = [int(indexAfterBatch)]
  print(dictInstances)
  return dictInstances
