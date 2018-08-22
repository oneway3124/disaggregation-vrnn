from __future__ import division, print_function, absolute_import
import dataSet_ts as dt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import datetime
import sys, time
############### HYPERPARAMETERS ################
lr = 0.05
#n_classes # calculated in reader.load_csvdata

nilmkt_fileName = sys.argv[1] # Path to ukdale.h5
n_steps = int(sys.argv[2]) # Size of input sequence
n_hidden_units = int(sys.argv[3]) #100   # neurons in hidden layer
training_iters = int(sys.argv[4])
stride_train = int(sys.argv[5])
stride_test =  int(sys.argv[6]) ### Until we can implement the avarage
n_output_units = 60
flg_relu  = 1
flg_outFF = 1
period = 6

# Setting the number of nodes in hidden states
n_inputs   = 1
n_inputs_0 = 100   # Number of cells in input vector
n_inputs_1 = 20
n_inputs_2 = n_steps

#Until 005 inclusive hu=100 o=20
#from 008 hu=200 o=50
#009 hu200 o50
#010 hu300 o60


########### FUNCTIONS TO DEFINE WEIGHTS AND BIAS AND CNN POOLS ####################

def weight_variable(shape, m=0, std=0.05):
    initial = tf.random_normal(shape,mean=m, stddev=std) #truncated_normal, stddev=0.1 / w['in'] = tf.random_normal
    return tf.Variable(initial)

def bias_variable(shape, c=1.0):
    initial = tf.constant(c) #, shape=shape
    return tf.Variable(initial)

def conv1d(x, W, s):
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv1d(x, W, stride=s, padding='SAME')

def max_pool_2x2(x, ksizeHeight):
    # stride [1, height_movement, width_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,ksizeHeight,1,1], strides=[1,ksizeHeight,1,1], padding='SAME')

def avg_pool_2x2(x, ksizeHeight):
    # stride [1, height_movement, width_movement, 1]
    return tf.nn.avg_pool(x, ksize=[1,ksizeHeight,1,1], strides=[1,ksizeHeight,1,1], padding='SAME')


############ UPLOADING DATA SET ############

reader = dt.ReaderTS(n_inputs, stride_train, stride_test)
XdataSet, YdataSet,n_classes = reader.load_csvdata(nilmkt_fileName,period,n_steps, pval_size=0.25, ptest_size=0.25)

x_train, x_test, y_train, y_test = np.transpose(XdataSet['train'],[0,2,1]),np.transpose(XdataSet['test'],[0,2,1]),np.transpose(YdataSet['train'],[0,2,1]),np.transpose(YdataSet['test'],[0,2,1])
x_val, y_val = np.transpose(XdataSet['val'],[0,2,1]), np.transpose(YdataSet['val'],[0,2,1])

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_classes])
# lens = tf.placeholder(tf.int32, [None]) # for RNN different lengths of instances
initBatch = tf.placeholder(tf.int32, shape=())


arrfieldnames = np.array(['training_iters', 'n_hidden_units', 'train error','testError'])

########## DEFINING  HIDDEN LAYERS ##########

Win_0 = weight_variable([n_steps,n_inputs_0])
bin_0 = bias_variable([n_inputs_0])
aux_0 = tf.transpose(x, perm=[0, 2, 1])
aux_0 = tf.reshape(aux_0, [initBatch * n_inputs, n_steps])
h_0 = tf.tanh(tf.matmul(aux_0, Win_0) +  bin_0) #just added tanh

Win_1 = weight_variable([n_inputs_0,n_inputs_1])
bin_1 = bias_variable([n_inputs_1])
h_1 = tf.tanh(tf.matmul(h_0, Win_1) +  bin_1)#just added tanh

Win_2 = weight_variable([n_inputs_1,n_inputs_2])
bin_2 = bias_variable([n_inputs_2])
h_2 = tf.tanh(tf.matmul(h_1, Win_2) +  bin_2) #just added tanh
h_2 = tf.reshape(h_2, [initBatch, n_inputs_2, n_inputs])

W_conv = weight_variable([int(n_inputs_2/10),n_inputs,n_classes])
b_conv = bias_variable([n_classes]) #,c=10.0
output = conv1d(h_2, W_conv, 1) + b_conv

tf.summary.histogram('Wout', W_conv)
tf.summary.histogram('h_1', h_1)
tf.summary.histogram('activations', output)

############ OPTIMIZER  AND COST FUNCTION DEFINITION ############

# Reshape output of CNN in order to apply sigmoid to each time step (of each instance) through all the appliances
predicRates0 = tf.reshape(tf.nn.relu(output),[initBatch * n_steps,n_classes])
predicRates1 = tf.nn.softmax(predicRates0)
# Sigmoid not necessary because we just want to have probabilities that represent percentages of the total consumpmtion
# Besides sigmoid just printed 0s

predicRates3= tf.reshape(predicRates1,[initBatch, n_steps,n_classes])
x_Original3 = tf.tile(x,[1,1,n_classes])
predicTotal_0 = tf.multiply(predicRates3, x_Original3) # Multiplying percentage by X to get predicted consumption of an appliance

predicTotal_1 = tf.reduce_sum(predicTotal_0, axis=2) # getting the total predicted aggregated
ySumUp = tf.reduce_sum(y,axis=2)

cost = tf.losses.mean_squared_error(ySumUp,predicTotal_1) # bydefault  reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
print("Cost: ",cost.shape)
l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.5)#scale=0.05, scope=None
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [W_conv])
train_op = tf.train.AdamOptimizer(lr, epsilon=0.1).minimize(cost+regularization_penalty*100000000)

# Getting the absolute difference between real values and predidcted ones for each appliance at each time step
mae0 = tf.reduce_mean(tf.abs(y-predicTotal_0),axis=0)
mae1 = tf.reduce_mean(mae0,axis=0) # Getting average of MAE

ySumUp_time = tf.reduce_sum(tf.reduce_sum(y,axis=1), axis=0 )
predSumUp_time = tf.reduce_sum(tf.reduce_sum(predicTotal_0, axis=1), axis=0 )
num = tf.abs(ySumUp_time - predSumUp_time)
correctly_assigned = num/tf.maximum(ySumUp_time,predSumUp_time) ## I guess it should be ySumUp_time
#avg_correcAssigned = tf.reduce_mean(correctly_assigned,axis=0)

print (output.shape)
print (predicRates1.shape)
print (predicTotal_0.shape)
print (ySumUp.shape)
tf.summary.histogram('preds', predicTotal_1)
tf.summary.scalar('cost', cost)
tf.summary.scalar('l1_reg', regularization_penalty)
tf.summary.histogram('mae1', mae1)

'''
trying to use contrib.lear that is supposed to be like sklearn (used in NEURALNILM)

target = tf.contrib.layers.flatten(y[:,:,0])
output2 = tf.contrib.layers.flatten(predicTotal_0[:,:,0])
mae_contrib = tf.contrib.metrics.streaming_mean_absolute_error(output2, target)
'''
# Show the sizes of the weight matrices to be able to know 
# how many paremeters the model is working with
for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(v)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    # Write values of some variables to check them later in tensorboard
    train_writer = tf.summary.FileWriter('Logs/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M"),sess.graph)
    test_writer = tf.summary.FileWriter('Logs/LogsTestAE/')
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while (step < training_iters):
    	[summary,op_train] = sess.run([merged, train_op], feed_dict={x: x_train,y: y_train,initBatch: x_train.shape[0]})
    	train_writer.add_summary(summary, step)
        # Print metrics on validation set every 1 of 10 partions.
        if step % (training_iters/10) == 0:
            trncost,metric,subMetric,predRat0,predRat1, predRat3, tot0, x3 = sess.run([cost,mae1,mae0, predicRates0,predicRates1,predicRates3,predicTotal_0, x_Original3], feed_dict={x: x_val, y: y_val, initBatch: x_val.shape[0]})
            #print("X {} X3 {}".format(x_train[0,292,:], x3[0,292,:]))
            print("Time{} Step {}    Cost {} Metric {} ".format(datetime.datetime.now(),step,trncost, metric, predRat0[292], predRat1[292], predRat3[0,292,:], tot0[0,292,:]))
            #beforePercentage {} Percentage {} back {} and multiplying by x {}
        step+=1
    tstcost, preds, metric, subMetric, p_num, p_ySumUp_time, p_predSumUp_time, corrAsig =sess.run([cost, predicTotal_0, mae1,mae0, num, ySumUp_time,predSumUp_time,  correctly_assigned], feed_dict={x: x_test, y: y_test,initBatch: x_test.shape[0]})
    print ("Test {}".format(tstcost))
    print ("Metric {} {}".format(metric, corrAsig.shape)) #,avg_corrAssig.shape, avg_corrAssig
    print ("Corr assigned {}".format(corrAsig[0:3]))
    print ("Predicted {}".format(p_predSumUp_time[0:3]))
    print ("Numerator {}".format( p_num[0:3]))
    print ("Denom {}".format( p_ySumUp_time[0:3]))
    
    #  results on test set, as they are ordered they can just
    newTest =np.concatenate([y_test[i] for i in range(y_test.shape[0])],0) # tf.concat([y_test[i] for i in range(y_test.shape[0])],0)
    newPred = np.concatenate([preds[i]  for i in range(y_test.shape[0])],0) #tf.concat([preds[i] for i in range(preds.shape[0])],0)   
    newInput = np.concatenate([x_test[i]  for i in range(y_test.shape[0])],0)
    print (newTest.shape, newPred.shape, type(newTest), type(newPred))

    plt.figure(1)
    plt.plot(newTest)
    plt.figure(2)
    plt.plot(newPred)
    plt.figure(3)
    plt.plot(newInput)
    #plt.figure(4)
    #plt.plot(avg_corrAssig)
    plt.show()
    '''
    print (y_test[0,:,:].shape, preds[0].shape, type(y_test[0,:,:]), type(preds[0]))
    plt.figure(1)
    plt.plot(y_test[0,:,:])
    plt.figure(2)
    plt.plot(preds[0])
    plt.show()
    '''