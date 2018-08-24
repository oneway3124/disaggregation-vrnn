from __future__ import division
import os
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import fnmatch
import re
import dataSet_ts as dt
from lxml import etree
import pickle


def fetch_redd(data_path, windows, appliances, numApps, period, n_steps, stride_train, stride_test, trainPer=0.5, valPer=0.25, testPer=0.25, typeLoad=0, flgAggSumScaled=0, flgFilterZeros=0,
                                              seq_per_batch=5000, target_inclusion_prob=0.5):
    '''
    Deleting huge part of seems like generating data from other metadata
    '''
    reader = dt.ReaderTS(windows, appliances, n_steps, stride_train, stride_test, period, flgAggSumScaled, flgFilterZeros,
                        flgScaling=0, trainPer=trainPer, valPer=valPer, testPer=testPer)


    if (numApps==-1):
        truFileName='all_'+str(n_steps)+'_'+str(period)+'_tr'+str(trainPer)+'_te'+str(testPer)+'_te'+str(valPer)+'_b'+str(windows.keys())
    else:
        truFileName=appliances[numApps]+'_'+str(n_steps)+'_'+str(period)+'_tr'+str(trainPer)+'_te'+str(testPer)+'_te'+str(valPer)+'_b'
        if (typeLoad==1):
            truFileName = truFileName+'_b'+str(windows['train'].keys())#+'_'+str(windows[0][0])+'_'+str(windows[0][1])
        else:
            truFileName = truFileName+'_b'+str(windows.keys())
    try:
        split = pickle.load( open(data_path+"/pickles/"+truFileName+".pickle","rb"))
        return split['X_train'], split['Y_train'], split['X_val'], split['Y_val'], split['X_test'], split['Y_test'], reader
    except (OSError, IOError) as e:
        XdataSet, YdataSet = reader.load_csvdata(data_path, numApps,typeLoad,seq_per_batch, target_inclusion_prob)
    #shape before: batch, apps, steps
        x_train, x_test, x_val, y_train, y_test, y_val = XdataSet['train'],XdataSet['test'],XdataSet['val'], YdataSet['train'],YdataSet['test'],YdataSet['val']
        x_train, x_val, x_test = np.expand_dims(x_train,axis=2), np.expand_dims(x_val,axis=2), np.expand_dims(x_test,axis=2)
        if (numApps!=-1):
          y_train,y_val,y_test = np.expand_dims(y_train,axis=2), np.expand_dims(y_val,axis=2), np.expand_dims(y_test,axis=2)
        with open(data_path+"/pickles/"+truFileName+".pickle",'wb') as splitWrite:
            pickle.dump({'X_train':x_train,'Y_train':y_train,'X_val':x_val,'Y_val':y_val,'X_test':x_test,'Y_test':y_test},splitWrite)

    return  x_train, y_train, x_val, y_val, x_test, y_test, reader
