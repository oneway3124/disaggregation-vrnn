from __future__ import division, print_function, absolute_import
import dataSet_ts as dt
#import tflearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import datetime
import sys
#from tflearn.data_utils import to_categorical, pad_sequences
#from tflearn.datasets import imdb

# hyperparameters
lr = 0.01
#n_classes = 5# or 10 channels
n_inputs = 1   # Number of cells in input vector
nilmkt_fileName = sys.argv[1]#'/home/gissella/Documents/Research/Disaggregation/UK-DALE/ukdale.h5'
n_steps = int(sys.argv[2]) #200    # time steps
n_hidden_units = int(sys.argv[3])#100   # neurons in hidden layer
training_iters = int(sys.argv[4])
stride_train = int(sys.argv[5])
stride_test =  int(sys.argv[6]) ### Until we can implement the avarage
n_input_units = 64
period = 6

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) #, stddev=0.1 / w['in'] = tf.random_normal
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1) #, shape=shape
    return tf.Variable(initial)


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

varName = str(training_iters) + '_'+ str(n_hidden_units)
#### INPUT LAYER
'''
Win = weight_variable([n_steps,n_input_units])
bin = bias_variable([n_input_units])
aux_in = tf.transpose(x, perm=[0, 2, 1])
aux_in = tf.reshape(aux_in, [initBatch,n_steps*n_inputs])
input_v = tf.matmul(aux_in, Win) +  bin
input_v = tf.reshape(input_v, [initBatch,n_input_units,n_inputs])

### basic LSTM Cell with INPUT LAYER
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True, activation=tf.tanh)
_init_state = lstm_cell.zero_state(initBatch, dtype=tf.float32)
outputLstm, final_state = tf.nn.dynamic_rnn(lstm_cell, input_v, initial_state=_init_state, time_major=False)

'''
### basic LSTM Cell without INPUT LAYER
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True, activation=tf.tanh)
_init_state = lstm_cell.zero_state(initBatch, dtype=tf.float32)
outputLstm, final_state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=_init_state, time_major=False)

### Output layer
Wout = weight_variable([n_hidden_units,n_classes ])
bout = bias_variable([n_classes])
aux = tf.transpose(outputLstm, perm=[0, 2, 1])
aux = tf.reshape(aux,[initBatch*n_steps,n_hidden_units]) #when no input layer: initBatch*n_steps
output = tf.matmul(aux, Wout) +  bout
output = tf.reshape(output, [initBatch, n_steps,n_classes])#when no input layer: initBatch, n_steps,n_classes
#output = tf.transpose(output, perm=[0, 2, 1])

tf.summary.histogram('Wout', Wout)
tf.summary.histogram('activations', outputLstm)
#tf.summary.histogram('W_lstm', 'RNN/BasicLSTMCell/Linear/Matrix:0')

###  Optimizer  ###
cost = tf.losses.mean_squared_error(y,output)
tf.summary.scalar('cost', cost)
#metric = tf.metrics.mean_absolute_error(y, output)
print (cost.shape)
#cost = tf.reduce_mean(tf.metrics.mean_squared_error(y,output))
#cost1 = tf.nn.l2_loss(y-output) # this is just regularization
#cost = tf.reduce_mean(cost1)
#cost =  tf.contrib.seq2seq.sequence_loss(output, y)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

y_appliance = tf.transpose(y,[1,0,2])
pred_appliance = tf.transpose(output,[1,0,2])

mae0 = tf.reduce_mean(tf.abs(y-output),axis=0)
mae1 = tf.reduce_mean(mae0,axis=0)

for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(v)
'''
if (stride_test<n_steps):
    calculate_average()
'''

with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('LogsTrain/',
                                      sess.graph)
    test_writer = tf.summary.FileWriter('LogsTest/')
    tf.global_variables_initializer().run()
    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while (step < training_iters):
    	[summary, op_train] = sess.run([merged, train_op], feed_dict={x: x_train,y: y_train,initBatch: x_train.shape[0]})
    	train_writer.add_summary(summary, step)
        if step % (training_iters/10) == 0:
    		trncost,metric,subMetric = sess.run([cost,mae1,mae0], feed_dict={x: x_val, y: y_val, initBatch: x_val.shape[0]})
    		print("Time{} Step {}    Cost {} Metric {}".format(datetime.datetime.now(),step,trncost, metric))
    	step+=1
    tstcost, preds, metric,subMetric =sess.run([cost, output, mae1,mae0], feed_dict={x: x_test, y: y_test,initBatch: x_test.shape[0]})
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