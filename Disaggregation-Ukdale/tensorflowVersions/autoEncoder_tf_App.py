from __future__ import division, print_function, absolute_import
import dataSet_ts as dt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import datetime
import sys, time
import os, math

############### HYPERPARAMETERS ################

appliances = [ 'kettle','microwave', 'washing machine', 'dish washer' , 'fridge']#
windows = {1:("2013-02-27", "2015-02-27")}
#To specify more than one building, just add a key in the dict windows
#i.e. 2:("2013-02-27", "2013-04-27")

def main(args):

    lr = float(args['lr']) #0.0001
    nilmkt_fileName = args['nilmkt_fileName']       # Path to ukdale.h5
    n_steps = int(args['n_steps'])                  # Size of input sequence (length of instance)
    n_hidden_units = int(args['hiddenUnitsLayer1']) # First layer number of units
    training_iters = int(args['iterations'])        # Number of iterations or epochs
    stride_train = int(args['strideTrain'])         # Separation between instances of training set
    stride_test =  n_steps                          # Separation between instances of test set
    applianceTest =  int(args['applianceToTest'])   # Apliance (order number in the list) of the appliance to evaluate
    period = 6                                      # It is the period that most of other works use

    # Setting the number of nodes in hidden states of the 3 layers
    n_inputs_0 = n_hidden_units   # Number of cells in input vector
    n_inputs_1 = int(n_hidden_units/2)
    n_inputs_2 = n_hidden_units
    n_inputs_3 = n_steps


    ########### FUNCTIONS TO DEFINE WEIGHTS AND BIAS ####################

    def weight_variable(shape, m=0, std=0.5):
        initial = tf.random_normal(shape,mean=m, stddev=std) #truncated_normal, stddev=0.1 / w['in'] = tf.random_normal
        return tf.Variable(initial)

    def bias_variable(shape, c=1.0):
        initial = tf.constant(c, shape=shape)
        return tf.Variable(initial)

    def conv1d(x, W, s):
        return tf.nn.conv1d(x, W, stride=s, padding='SAME')

    ############ UPLOADING DATA SET ############
    reader = dt.ReaderTS(windows, appliances, n_steps, stride_train, stride_test, period,
                            flgScaling=1, trainPer=0.5, valPer=0.25, testPer=0.25)

    XdataSet, YdataSet = reader.load_csvdata(nilmkt_fileName, applianceTest) 
    # Depending on the applianceTest (-1 for all): x.shape = [batch, seqLen]
    #                                y.shape = [batch, seqLen] or [batch, seqLen, numAppliances]                    
    x_train, x_test, x_val  = XdataSet['train'], XdataSet['test'],  XdataSet['val']
    y_train, y_test, y_val  = YdataSet['train'], YdataSet['test'],  YdataSet['val']


    ##############   STARTING TO CONSTRUCT GRAPH: INPUTS AND OPERATIONS #####################

    x = tf.placeholder(tf.float32, [None, n_steps])
    y = tf.placeholder(tf.float32, [None, n_steps])
    # Size of the batch (it could change in validation and/or test runs)
    initBatch = tf.placeholder(tf.int32, shape=()) 

    ## Definint layers

    Win_0 = weight_variable([n_steps,n_inputs_0])
    bin_0 = bias_variable([n_inputs_0])
    h_0 = tf.sigmoid(tf.matmul(x, Win_0) +  bin_0)

    Win_1 = weight_variable([n_inputs_0,n_inputs_1])
    bin_1 = bias_variable([n_inputs_1])
    h_1 = tf.sigmoid(tf.matmul(h_0, Win_1) +  bin_1)

    Win_2 = weight_variable([n_inputs_1,n_inputs_2])
    bin_2 = bias_variable([n_inputs_2])
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, Win_2) +  bin_2)

    Win_3 = weight_variable([n_inputs_2,n_inputs_3])
    bin_3 = bias_variable([n_inputs_3])
    pred = tf.nn.relu(tf.matmul(h_2, Win_3) +  bin_3)

    ## Indicating to tensorflow that save the values of the parameters/layers in each iteration##
    tf.summary.histogram('Win_0', Win_0)
    tf.summary.histogram('Win_1', Win_1)
    tf.summary.histogram('Win_2', Win_2)

    ### Flatten real Y and prediction to be able to measure the results as other models do
    yForMetric = tf.reshape(y,[-1])
    predForMetric = tf.reshape(pred,[-1])

    ### Calculating metrics (done just in validation/testing. Not part of the optimization)
    #Check what's the difference btw these different calculatinos. Meanwhile, staying with upMae0
    #tf.metrics.mean_absolute_error. =  tf.contrib.metrics.streaming_mean_absolute_error
    mae0, upMae0 = tf.metrics.mean_absolute_error(labels=yForMetric,predictions=predForMetric) # Some problem with initialization
    upMae1 = tf.reduce_mean(tf.abs(yForMetric-predForMetric))

    sum_output = tf.reduce_sum(pred)
    sum_target = tf.reduce_sum(y)
    relativeError = (sum_output - sum_target) / tf.maximum(sum_output, sum_target)

    #### Defining cost function and optimizator 
    cost = tf.losses.mean_squared_error(yForMetric,predForMetric)
    tf.summary.scalar('cost', cost) #Track values of cost
    '''
    #Evaluate to add a regularizer
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.5)#scale=0.05, scope=None
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [W_conv])
    '''
    train_op = tf.train.AdamOptimizer(lr, epsilon=0.1).minimize(cost)

    # Show the sizes of the weight matrices to know number of total paramters used
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(v)

    ## Auxiliar structure to track cost and metrics. 
    ## Could be done w/ summary.scalar but haven't figured out yet - Tensorboard
    tracking = {'step': [], 'cost': [], 'mae': [], 'relerr':[]}

    ############   START RUNNING   #############

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tf.local_variables_initializer()) # for mean absolute error, that need local variables initialized
        merged = tf.summary.merge_all()
        # Write values of some variables to check them later in tensorboard
        train_writer = tf.summary.FileWriter('Logs/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M"),sess.graph)
        test_writer = tf.summary.FileWriter('Logs/LogsTestAE_SepApp/')
        # Print metrics on validation set every (1/10) of the iterations
        step = 0
        while (step < training_iters):
            [summary,op_train] = sess.run([merged, train_op], feed_dict={x: x_train,y: y_train,initBatch: x_train.shape[0]})
            train_writer.add_summary(summary, step)
            if(step % (training_iters/10) == 0):
                trncost, metric_upMae0, metric_upMae1, out_preds, out_re = sess.run(
                    [cost,upMae0, upMae1, pred, relativeError], feed_dict={x: x_val, y: y_val, initBatch: x_val.shape[0]})
                tracking['step'].append(step)
                tracking['cost'].append(trncost)
                tracking['mae'].append(metric_upMae0)
                tracking['relerr'].append(out_re)
                print("Time {} Step {:5d} Cost {:10.2f} Metric {:6.2f}/{:6.2f} RelError {:3.2f}".format(
                    datetime.datetime.now().strftime("%y-%m-%d_%H-%M"),step,trncost,metric_upMae0, metric_upMae1 ,out_re))
            step+=1
        tstcost, out_preds, metric_upMae0, out_re  =sess.run([cost, pred,upMae0, relativeError], feed_dict={x: x_test, y: y_test,initBatch: x_test.shape[0]})
        print ("\nTEST Cost {} Metric {} Relat err {}".format(tstcost,metric_upMae0, out_re))
        
        savedFolder = 'Experiments/app{}_{}'.format(applianceTest,datetime.datetime.now().strftime("%y-%m-%d_%H-%M"))
        os.makedirs(savedFolder)
        yFlat = y_test.flatten()
        predFlat = out_preds.flatten()

        # Concatenate results horizontally to have again the one large sequence
        newTest =np.concatenate([y_test[i] for i in range(y_test.shape[0])],0) # tf.concat([y_test[i] for i in range(y_test.shape[0])],0)
        newPred = np.concatenate([out_preds[i]  for i in range(y_test.shape[0])],0) #tf.concat([preds[i] for i in range(preds.shape[0])],0)   
        newInput = np.concatenate([x_test[i]  for i in range(y_test.shape[0])],0)

        assert newTest.shape == newPred.shape
        print ("Test size {}".format(newTest.shape))

        fLog = open('{}/{}_output'.format(savedFolder,applianceTest), 'w')
        fLog.write("Step,Cost,MAE, Rel error\n")
        for i , item in enumerate(tracking['step']):
          a = tracking['step'][i]
          b = tracking['cost'][i]
          c = tracking['mae'][i]
          d = tracking['relerr'][i]
          fLog.write("{},{},{},{}\n".format(a,b,c,d))
        fLog.write("Test,{},{},{}\n".format(tstcost,metric_upMae0,out_re))
        f, axarr = plt.subplots(3, 1, sharex=True)

        axarr[0].plot(newTest)
        axarr[0].set_title('Y-real')

        axarr[1].plot(newPred)
        axarr[1].set_title('Y-pred')

        axarr[2].plot(newInput)
        axarr[2].set_title('X-real')

        f.subplots_adjust(top=0.92, bottom=0.05, left=0.10, right=0.95, hspace=0.3, wspace=0.3)
        plt.savefig('{}/{}_plots'.format(savedFolder,applianceTest))#self.savedFolder+'/'+ch.name+str(numfig)
        plt.clf()

if __name__ == "__main__":

    import sys, time
    if len(sys.argv) > 1:
        config_file_name = sys.argv[-1]
    else:
        config_file_name = 'config.txt'

    f = open(config_file_name, 'r')
    lines = f.readlines()
    params = {}

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')
        param_name = param_list[0]
        param_value = param_list[1]
        params[param_name] = param_value

    main(params)
