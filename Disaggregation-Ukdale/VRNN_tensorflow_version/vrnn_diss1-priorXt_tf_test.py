import numpy as np
import tensorflow as tf
import argparse
import shutil
import os
import cPickle
import time

from datetime import datetime
from test_model_vrnn_dis1 import test_VRNN

from cle.cle.utils.compat import OrderedDict
from iterator import Iterator

import random


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def test(args, data):
	'''
	This function performs the testing on new data using the weights and biases from the trained model. It generates the prediction 
	using new model which is defines in test_VRNN. It finds the MSE and MAE for the predictions.

	Arguments
	args: The saved arguments of the trained model.
	data: test data (unseen data by the model)
	'''

	#directory where the trained model is saved
	dirname = 'save-vrnn'

	#testing data
	Xtest,ytest = data

	#Iterator object to split the data into batches
	test = Iterator(Xtest,ytest,batch_size = args.batch_size,n_steps=args.seq_length)
	n_batches = test.nbatches
	Xtest,ytest = test.get_split() 

	#create a new session to get the stored layers from trained model and also to run testing
	with tf.Session() as sess:

		old_params = []

		#import the trained model's graph into this session
		new_saver = tf.train.import_meta_graph('save-vrnn/'+args.appliance+'/final_model_'+str(args.num_epochs)+'_'+str(args.learning_rate)+'.ckpt.meta')
		new_saver.restore(sess, tf.train.latest_checkpoint('./save-vrnn/'+args.appliance+'/'))
 

		g1 = tf.get_default_graph()

		#layers from trained model whose weights and biases are required in the test model
		test_layers = ["rnn/VartiationalRNNCell/x_1/Linear/Matrix:0", "rnn/VartiationalRNNCell/x_1/Linear/bias:0", "rnn/VartiationalRNNCell/Prior/hidden/Linear/Matrix:0",
				"rnn/VartiationalRNNCell/Prior/hidden/Linear/bias:0", "rnn/VartiationalRNNCell/Prior/mu/Linear/Matrix:0", "rnn/VartiationalRNNCell/Prior/mu/Linear/bias:0",
				"rnn/VartiationalRNNCell/Prior/sigma/Linear/Matrix:0", "rnn/VartiationalRNNCell/Prior/sigma/Linear/bias:0", "rnn/VartiationalRNNCell/z_1/Linear/Matrix:0", 
				"rnn/VartiationalRNNCell/z_1/Linear/bias:0", "rnn/VartiationalRNNCell/Theta/hidden/Linear/Matrix:0", "rnn/VartiationalRNNCell/Theta/hidden/Linear/bias:0",
				"rnn/VartiationalRNNCell/Theta/mu/Linear/Matrix:0", "rnn/VartiationalRNNCell/Theta/mu/Linear/bias:0", "rnn/VartiationalRNNCell/Theta/sigma/Linear/Matrix:0",
				"rnn/VartiationalRNNCell/Theta/sigma/Linear/bias:0", "rnn/VartiationalRNNCell/Theta/coeff/Linear/Matrix:0", "rnn/VartiationalRNNCell/Theta/coeff/Linear/bias:0",
				"rnn/VartiationalRNNCell/y_1/Linear/Matrix:0", "rnn/VartiationalRNNCell/y_1/Linear/bias:0", "rnn/VartiationalRNNCell/lstm_cell/weights:0", "rnn/VartiationalRNNCell/lstm_cell/biases:0"]

		#get required tensors and variables for the test model
		for layer in test_layers:
			tensor = g1.get_tensor_by_name(layer)
			old_params.append(tensor)


		print("Loaded Model Weights and Biases!")

		#create an object of test model
		model = test_VRNN(args)

		#get the trainable variables from the graph
		trainable = tf.trainable_variables()

		#The last 22 variables in all belong to the test model (rest are from the train model that was imported)
		trainable = trainable[-22:]

		num_train_var = len(trainable)

		#assign the weight and bias tensors in test model
		for i in range(num_train_var):
			assign_op = trainable[i].assign(old_params[i])
			sess.run(assign_op)

		print("Assigned weights to Test Model")
	
		mae = []
		mse = []

		timesteps = [ x for x in range(1,201)]

		#run the testing through all the batches in the test data
	    	for b in xrange(n_batches):
		        x = Xtest[b]
			feed = {model.input_x: x} #input to test model
			pred = sess.run(model.pred,feed)

			#reshape the predicted data to [batch_size, timesteps, n_app] size
			pred = np.reshape(pred, [250,200,-1]) 
			label = np.array(ytest[b]).astype(float)
			pred = np.array(pred).astype(float)

			#compute the mse and mae values on the predictions
			mae_i  = np.reshape(np.absolute((label - pred)),[-1,]).mean()
			mse_i =  np.reshape((label - pred)**2,[-1,]).mean()
			mae.append(mae_i)
			mse.append(mse_i)

			plt.plot(timesteps, np.reshape(pred[0],(-1,)))
			plt.plot(timesteps, np.reshape(np.reshape(x,(250,200,-1))[0],(-1,)))
			plot.show()
			break

		print("MAE:",sum(mae)/len(mae))
		print("MSE:",sum(mse)/len(mse))

if __name__ == "__main__":

	#load the saved args from the appropriate folder
	'''eg: if the trained model is for kettle then the trained model is saved in save-vrnn/kettle
		replace below as os.path.join('save-vrnn/kettle', 'config.pkl') '''
	with open(os.path.join('save-vrnn/fridge', 'config.pkl')) as f: #should define the path properly
    		saved_args = cPickle.load(f)

	data = np.load("testData_"+saved_args.appliance+".npy") #load the test data that was preprocessed data and saved

	Xtest = data[0]
	ytest = data[1]

	test(saved_args, (Xtest,ytest))
