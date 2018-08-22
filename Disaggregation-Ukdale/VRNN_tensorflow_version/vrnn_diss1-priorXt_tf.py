import numpy as np
import tensorflow as tf
import argparse
import shutil
import os
import cPickle
import time

from datetime import datetime
from model_vrnn_dis1 import VRNN

from VRNN_theano_version.datasets.ukdale_utils import fetch_ukdale
from cle.cle.utils.compat import OrderedDict
from iterator import Iterator

import random

appliances = [ 'kettle','microwave', 'washing machine', 'dish washer' , 'fridge'] #appliances in the dataset considered

#range of dates to split the data into train, test and validation
windows={'train': {1: ("2013-02-27", "2015-02-27")}, 'test':{1:("2015-02-27", "2016-02-27")}, 'val':{1:("2016-02-27", "2017-02-27")}}

def train(args, model,data):
    dirname = 'save-vrnn/'+args.appliance
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    ckpt = tf.train.get_checkpoint_state(dirname) #check if there exists a previously trained model in the checkpoint

    Xtrain,ytrain = data
    train = Iterator(Xtrain,ytrain,batch_size = args.batch_size,n_steps=args.seq_length) #to split data into batches
    n_batches = train.nbatches
    Xtrain,ytrain = train.get_split()
    mae = []
    mse = []
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
        check = tf.add_check_numerics_ops()
        merged = tf.summary.merge_all()
        tf.global_variables_initializer().run() #initialize all variables in the graph as defined
        saver = tf.train.Saver(tf.global_variables())
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path) #restore previously saved model
            print "Loaded model"
        start = time.time()
	state_c = None
	state_h = None
        for e in xrange(args.num_epochs):
            #assign learning rate 
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e))) 

            #get the initial state of lstm cell 
            state = model.initial_state_c, model.initial_state_h
            mae.append([])
            mse.append([])
            for b in xrange(n_batches):
                x = Xtrain[b]
                y = ytrain[b]
                feed = {model.input_x: x, model.input_y: y, model.target_data: y} # input data : x and y ; target data : y

                #train the model on this batch of data
                train_loss, _, cr, summary, sigma, mu, inp, target, state_c, state_h, pred = sess.run(
                        [model.cost, model.train_op, check, merged, model.sigma, model.mu, model.flat_input, model.target, model.final_state_c, model.final_state_h, model.output],
                                                             feed)

                summary_writer.add_summary(summary, e * n_batches + b)

		#the output from the model is in the shape [50000,1] reshape to 3D (batch_size, time_steps, n_app)
                pred = np.array(np.reshape(pred, [250,200,-1])).astype(float)
                label = np.array(y).astype(float)

		#compute mae and mse for the output
                mae_i  = np.reshape(np.absolute((label - pred)),[-1,]).mean()
                mse_i =  np.reshape((label - pred)**2,[-1,]).mean()

                mae[e].append(mae_i)
                mse[e].append(mse_i)

		#save the model after every 800 (monitoring_freq) epochs
                if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                    checkpoint_path = os.path.join(dirname, 'model_'+str(args.num_epochs)+'_'+str(args.learning_rate)+'.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                    print "model saved to {}".format(checkpoint_path)

                end = time.time()
		
                print "{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}, std = {:.3f}" \
                    .format(e * n_batches + b,
                            args.num_epochs * n_batches,
                            e, args.chunk_samples * train_loss, end - start, sigma.mean(axis=0).mean(axis=0))
                start = time.time()

            #the average mae,mse values in every epoch
            print "Epoch {}, mae = {:.3f}, mse = {:.3f}".format(e, sum(mae[e])/len(mae[e]), sum(mse[e])/len(mse[e]))

	#path to save the final model
	checkpoint_path = os.path.join(dirname, 'final_model_'+str(args.num_epochs)+'_'+str(args.learning_rate)+'.ckpt') 

	saver2 = tf.train.Saver()
	saver2.save(sess, checkpoint_path)

	print "model saved to {}".format(checkpoint_path)
			
def main(args):

	'''

	Arguments:
	args : A dictionary containing all parameters required for the training ( read from config_AE.txt ).


	Returns: None
	'''
	trial = int(args['trial'])
	pkl_name = 'vrnn_gmm_%d' % trial
	channel_name = 'mse'

	data_path = args['data_path']
	save_path = args['save_path'] 
	flgMSE = int(args['flgMSE'])

	genCase = int(args['genCase'])
	period = int(args['period'])
	n_steps = int(args['n_steps'])
	stride_train = int(args['stride_train'])
	stride_test = n_steps #int(args['stride_test'])

	monitoring_freq = int(args['monitoring_freq'])
	epoch = int(args['epoch'])
	batch_size = int(args['batch_size'])
	x_dim = int(args['x_dim'])
	y_dim = int(args['y_dim'])
	flgAgg = int(args['flgAgg'])
	z_dim = int(args['z_dim'])
	rnn_dim = int(args['rnn_dim'])
	k = int(args['num_k']) #a mixture of K Gaussian functions
	lr = float(args['lr'])
	debug = int(args['debug'])
	num_sequences_per_batch = int(args['numSequences']) #based on appliance
	typeLoad = int(args['typeLoad'])
	target_inclusion_prob = float(args['target_inclusion_prob'])

	print "-------------------------------------------------------------------\n"
	print "trial no. %d" % trial
	print "learning rate %f" % lr
	print "saving pkl file '%s'" % pkl_name
	print "to the save path '%s'" % save_path
	print "Appliance '%s'" % appliances[flgAgg]
	print "Epochs: %d" % epoch
	print "-------------------------------------------------------------------\n"

	n_appl = 1 # number of appliance or the size of 3rd dimension

	#Data Preprocessing

	Xtrain, ytrain, Xval, yval, Xtest, ytest, reader = fetch_ukdale(data_path, windows, appliances,numApps=flgAgg, period=period,
  n_steps= n_steps, stride_train = stride_train, stride_test = stride_test,
  typeLoad= typeLoad, flgAggSumScaled = 1, flgFilterZeros = 1,
  seq_per_batch=num_sequences_per_batch, target_inclusion_prob=target_inclusion_prob)

	#creating a argument parser for the model
	parser = argparse.ArgumentParser()
	parser.add_argument('--rnn_size', type=int, default=rnn_dim, 
			help='size of RNN hidden state')
	parser.add_argument('--latent_size', type=int, default=z_dim,
			help='size of latent space')
	parser.add_argument('--batch_size', type=int, default=batch_size,
			help='minibatch size')
	parser.add_argument('--seq_length', type=int, default=n_steps,
                        help='RNN sequence length')
	parser.add_argument('--num_epochs', type=int, default=epoch,
                        help='number of epochs')
	parser.add_argument('--save_every', type=int, default=monitoring_freq,
                        help='save frequency')
   	parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
	parser.add_argument('--learning_rate', type=float, default=lr,
                        help='learning rate')
	parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
	parser.add_argument('--chunk_samples', type=int, default=n_appl,
                        help='number of samples per mdct chunk')
	parser.add_argument('--target_dim', type=int, default=k,
                        help='mixture of K Gaussian functions')
	parser.add_argument('--appliance', type=str, default=appliances[flgAgg],
                        help='which appliance data to use')
	args = parser.parse_args()

	model = VRNN(args) #create the VRNN model

	states = train(args, model,(Xtrain,ytrain))
	
	#save the preprocessed test data
	save_file = "testData_"+str(args.appliance)+".npy"
	if not os.path.exists('./'+save_file):
		np.save("testData_"+str(args.appliance)+".npy",np.array([Xtest,ytest]))

if __name__ == "__main__":

	import sys, time
	if len(sys.argv) > 1:
		config_file_name = sys.argv[-1]
	else:
		config_file_name = 'config_AE.txt'

	f = open(config_file_name, 'r')
	lines = f.readlines()
	params = OrderedDict()

	for line in lines:
		line = line.split('\n')[0]
		param_list = line.split(' ')
		param_name = param_list[0]
		param_value = param_list[1]
		params[param_name] = param_value

	params['save_path'] = params['save_path']+'/gmmAE/'+datetime.now().strftime("%y-%m-%d_%H-%M")+'_app'+params['flgAgg']
	os.makedirs(params['save_path'])
	shutil.copy('config_AE.txt', params['save_path']+'/config_AE.txt')

	main(params)
