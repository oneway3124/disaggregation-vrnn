from __future__ import division, print_function, absolute_import
import dataSet_ts as dt
#import tflearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import datetime
import sys
from neuralnilm.metrics import Metrics
import sklearn.metrics as metrics



#from tflearn.data_utils import to_categorical, pad_sequences
#from tflearn.datasets import imdb

# hyperparameters
lr = 100
#n_classes = 5# or 10 channels

nilmkt_fileName = sys.argv[1]#'/home/gissella/Documents/Research/Disaggregation/UK-DALE/ukdale.h5'
n_steps = int(sys.argv[2]) #200    # time steps
n_hidden_units = int(sys.argv[3])#100   # neurons in hidden layer
training_iters = int(sys.argv[4])
stride_train = int(sys.argv[5])
stride_test =  int(sys.argv[6]) ### Until we can implement the avarage
flg_relu  = 1
flg_outFF = 1
period = 6

n_channels = 1
n_inputs   = 1
n_inputs_0 = 10   # Number of cells in input vector
n_inputs_1 = 3
n_inputs_2 = stride_test

#Until 005 inclusive hu=100 o=20
#from 008 hu=200 o=50
#009 hu200 o50
#010 hu300 o60

def weight_variable(shape, m=0, std=1.0):
    initial = tf.random_normal(shape,mean=m, stddev=std) #truncated_normal, stddev=0.1 / w['in'] = tf.random_normal
    return tf.Variable(initial)

def bias_variable(shape, c=1.0):
    initial = tf.constant(c) #, shape=shape
    return tf.Variable(initial)

def conv1d(x, W, s):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv1d(x, W, stride=s, padding='SAME')

def max_pool_2x2(x, ksizeHeight):
    # stride [1, height_movement, width_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,ksizeHeight,1,1], strides=[1,ksizeHeight,1,1], padding='SAME')

def avg_pool_2x2(x, ksizeHeight):
    # stride [1, height_movement, width_movement, 1]
    return tf.nn.avg_pool(x, ksize=[1,ksizeHeight,1,1], strides=[1,ksizeHeight,1,1], padding='SAME')



# DATAPORT uploading
reader = dt.ReaderTS(n_inputs, stride_train, stride_test)
XdataSet, YdataSet,n_classes = reader.load_csvdata(nilmkt_fileName,period,n_steps)

x_train, x_test, y_train, y_test = np.transpose(XdataSet['train'],[0,2,1]),np.transpose(XdataSet['test'],[0,2,1]),np.transpose(YdataSet['train'],[0,2,1]),np.transpose(YdataSet['test'],[0,2,1])
x_val, y_val = np.transpose(XdataSet['val'],[0,2,1]), np.transpose(YdataSet['val'],[0,2,1])

#print x_train.shape, x_test.shape, y_train.shape, y_test.shape

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_classes])
lens = tf.placeholder(tf.int32, [None])
initBatch = tf.placeholder(tf.int32, shape=())


arrfieldnames = np.array(['training_iters', 'n_hidden_units', 'train error','testError'])

#### HIDDEN LAYERS

Win_0 = weight_variable([n_steps,n_inputs_0])
bin_0 = bias_variable([n_inputs_0])
aux_0 = tf.transpose(x, perm=[0, 2, 1])
aux_0 = tf.reshape(aux_0, [initBatch * n_inputs, n_steps])
h_0 = tf.matmul(aux_0, Win_0) +  bin_0

Win_1 = weight_variable([n_inputs_0,n_inputs_1])
bin_1 = bias_variable([n_inputs_1])
h_1 = tf.matmul(h_0, Win_1) +  bin_1

Win_2 = weight_variable([n_inputs_1,n_inputs_2])
bin_2 = bias_variable([n_inputs_2])
h_2 = tf.matmul(h_1, Win_2) +  bin_2
h_2 = tf.reshape(h_2, [initBatch, n_inputs_2, n_inputs])

W_conv = weight_variable([int(n_inputs_2/10),n_inputs,n_classes], std=0.1)
b_conv = bias_variable([n_classes]) #,c=10.0
output = conv1d(h_2, W_conv, 1) + b_conv

tf.summary.histogram('Wout', W_conv)
tf.summary.histogram('h_1', h_1)
tf.summary.histogram('activations', output)

###  Optimizer  ###
print (output.shape)
#predicRates0 = tf.nn.tanh(output)
predicRates0= tf.reshape(output,[initBatch * n_steps,n_classes])
predicRates1 = tf.nn.sigmoid(predicRates0) # With dim=2 seems like it was dividing the whole steps*appliances matrix
predicRates2 = tf.nn.softmax(predicRates1)
### when printing no relu - softmax depending on the dimension I get all 0s or very little numbers that do not sum 1
# Something might be wrong in the application of the function over the dimensions
print (predicRates2.shape)
predicRates3= tf.reshape(predicRates2,[initBatch, n_steps,n_classes])
predicTotal_0 = tf.multiply(predicRates3,x)
print (predicTotal_0.shape)
predicTotal_1 = tf.reduce_sum(predicTotal_0, axis=2)
predicTotal_2= tf.reshape(predicTotal_1,[initBatch, n_steps,n_inputs])
predicTotal_3 = tf.nn.relu(predicTotal_2)
tf.summary.histogram('preds', predicTotal_2)

ySumUp = tf.reduce_sum(y,axis=2)
print (ySumUp.shape)
ySumUp = tf.reshape(ySumUp,[initBatch, n_steps,n_inputs])
cost = tf.losses.mean_squared_error(ySumUp,predicTotal_3)
tf.summary.scalar('cost', cost)

train_op = tf.train.AdamOptimizer(lr).minimize(cost)

mae0 = tf.reduce_mean(tf.abs(y-predicTotal_0),axis=0)
mae1 = tf.reduce_mean(mae0,axis=0)
tf.summary.histogram('mae1', mae1)

### Trying to use same metrics as NEURALNILM
target = tf.contrib.layers.flatten(y[:,:,0])
output2 = tf.contrib.layers.flatten(predicTotal_0[:,:,0])
metrics=Metrics(state_boundaries=[2.5])
print (target.shape, output2.shape)
if output.shape != target.shape:
    print ("WTF")
#scores = metrics.compute_metrics(output2,target)

#trying to use contrib.lear that is supposed to be like sklearn (used in NEURALNILM)
'''
target = tf.contrib.layers.flatten(y[:,:,0])
output2 = tf.contrib.layers.flatten(predicTotal_0[:,:,0])
mae_contrib = tf.contrib.metrics.streaming_mean_absolute_error(output2, target)
'''

for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(v)

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('LogsTrainAE/',
                                      sess.graph)
    test_writer = tf.summary.FileWriter('LogsTestAE/')
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while (step < training_iters):
    	[summary,op_train] = sess.run([merged, train_op], feed_dict={x: x_train,y: y_train,initBatch: x_train.shape[0]})
    	train_writer.add_summary(summary, step)
        if step % (training_iters/10) == 0:
    		trncost,metric,subMetric,predRat0,predRat1, predRat2 = sess.run([cost,mae1,mae0, predicRates0,predicRates1,predicRates2], feed_dict={x: x_val, y: y_val, initBatch: x_val.shape[0]})
    		print("Time{} Step {}    Cost {} Metric {} Percentage".format(datetime.datetime.now(),step,trncost, metric, predRat0[0],predRat1[0], predRat2[0]))
    	step+=1
    tstcost, preds, metric, subMetric =sess.run([cost, predicTotal_0, mae1,mae0], feed_dict={x: x_test, y: y_test,initBatch: x_test.shape[0]})
    print ("Test {}".format(tstcost))
    print ("Metric {}".format(metric))
    
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
    plt.show()
    '''
    print (y_test[0,:,:].shape, preds[0].shape, type(y_test[0,:,:]), type(preds[0]))
    plt.figure(1)
    plt.plot(y_test[0,:,:])
    plt.figure(2)
    plt.plot(preds[0])
    plt.show()
    '''

'''
# Network building
net = tflearn.input_data([None,n_inputs ,n_steps])
net = tflearn.lstm(net, n_hidden_units) #, dropout=0.8
#net = tflearn.fully_connected(net,n_classes, activation='linear')

net.sequence_loss(logits, targets, weights)
y_placeHolder = tf.placeholder(shape=[None, n_classes, n_steps], dtype=tf.float32)
net = tflearn.regression(net, placeholder=y_placeHolder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square')

generate (seq_length, temperature=0.5, seq_seed=None, display=False)

# Training
#model = tflearn.DNN(net, tensorboard_verbose=0)
model = tflearn.models.generator.SequenceGenerator(net, dictionary=dict(enumerate(np.arange(0,1,0.01))), seq_maxlen=n_steps, clip_gradients=0.0, tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/', checkpoint_path=None, max_checkpoints=None, session=None)
model.fit(x_train, y_train,  n_epoch=500, validation_set=(x_val, y_val), show_metric=True)


predictions = model.predict(x_test) 
print(predictions)
print(predictions.shape)#(1179, 10)


fig, ax = plt.subplots(1)
fig.autofmt_xdate()

#Selecting just the first
plot_predicted, = ax.plot(predictions[0:100,0], label='prediction')

plot_test, = ax.plot(y_test[0:100,0], label='Real value')

# ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d %H')
plt.title('Barcelona water prediction - 1st DMA')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
'''