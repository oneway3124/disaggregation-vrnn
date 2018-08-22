import numpy as np
import tensorflow as tf
import argparse
import shutil
import os
import cPickle
import time

from datetime import datetime
from model_vrnn_disall import VRNN

from preprocessing.ukdale_utils import fetch_ukdale
from cle.cle.utils.compat import OrderedDict
from iterator import Iterator


import random
import csv

import pandas as pd##

appliances = [ 'kettle','microwave', 'washing machine', 'dish washer' , 'fridge'] #appliances in the dataset considered

#range of dates to split the data into train, test and validation
windows = {1:("2015-02-01", "2017-02-01")}

def train(args, model,data,val_data):
    dirname = 'save-vrnn/'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    ckpt = tf.train.get_checkpoint_state(dirname) #check if there exists a previously trained model in the checkpoint

    Xtrain,ytrain = data
    Xval, yval = val_data


    shape1 = np.shape(Xtrain)
    df1 = pd.DataFrame(np.reshape(Xtrain,(shape1[0],-1)))
    shape2 = np.shape(ytrain)
    df2 = pd.DataFrame(np.reshape(ytrain,(shape2[0],-1)))
    print("\nXtrain")
    print(df1.describe())
    print('\nytrain')
    print(df2.describe())

    train = Iterator(Xtrain,ytrain,batch_size = args.batch_size,n_steps=args.seq_length,shape_diff=True) #to split data into batches
    n_batches = train.nbatches
    Xtrain,ytrain = train.get_split()

    

    #split validation data into batches
    validate = Iterator(Xval,yval,batch_size = args.batch_size,n_steps=args.seq_length,shape_diff=True)
    val_nbatches = validate.nbatches
    Xval, yval = validate.get_split()

    myFile = open(dirname+'/outputValidation.csv', 'w')
    writer = csv.writer(myFile)
    writer.writerows([["Epoch","Train_Loss","MAE","MSE"]])

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

        logs = [] 
        for e in xrange(args.num_epochs):
            #assign learning rate 
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e))) 

            #get the initial state of lstm cell 
            state = model.initial_state_c, model.initial_state_h
            mae.append([])
            mse.append([])

            prior_mean = [] ##
            phi_mean = [] ##
            if((e+1)%10 != 0):
		    for b in xrange(n_batches):
		        x = Xtrain[b]
		        y = ytrain[b]
		        feed = {model.input_x: x, model.input_y: y, model.target_data: y} # input data : x and y ; target data : y

		        #train the model on this batch of data
		        train_loss, _, cr, summary, sigma, mu, inp, target, state_c, state_h, pred, prior_mu, phi_mu = sess.run(
		                [model.cost, model.train_op, check, merged, model.sigma, model.mu, model.flat_input, model.target, model.final_state_c, model.final_state_h, model.output, model.prior_mu, model.phi_mu], feed) ##

                        prior_mean.append(prior_mu)  ##
                        phi_mean.append(phi_mu)   ##

		        summary_writer.add_summary(summary, e * n_batches + b)

			pred = np.concatenate(pred, axis=1)
			sigma = np.concatenate(sigma, axis=1)
			mu = np.concatenate(mu, axis=1)

			#the output from the model is in the shape [50000,1] reshape to 3D (batch_size, time_steps, n_app)
		        pred = np.array(np.reshape(pred, [args.batch_size,args.seq_length,-1])).astype(float)
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
            else: #pass validation data
		print("\nValidation Data\n")
		loss = 0
		for b in xrange(val_nbatches):
		        x = Xval[b]
		        y = yval[b]
		        feed = {model.input_x: x, model.input_y: y, model.target_data: y} # input data : x and y ; target data : y

		        #train the model on this batch of data
		        train_loss, cr, summary, sigma, mu, inp, target, state_c, state_h, pred = sess.run(
		                [model.cost, check, merged, model.sigma, model.mu, model.flat_input, model.target, model.final_state_c, model.final_state_h, model.output],
		                                                     feed)
		        loss += train_loss
		        summary_writer.add_summary(summary, e * n_batches + b)

			pred = np.concatenate(pred, axis=1)
			sigma = np.concatenate(sigma, axis=1)
			mu = np.concatenate(mu, axis=1)

			#the output from the model is in the shape [50000,1] reshape to 3D (batch_size, time_steps, n_app)
		        pred = np.array(np.reshape(pred, [args.batch_size,args.seq_length,-1])).astype(float)
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
		logs.append([e,train_loss/val_nbatches,sum(mae[e])/len(mae[e]), sum(mse[e])/len(mse[e])])

            #the average mae,mse values in every epoch
            print "Epoch {}, mae = {:.3f}, mse = {:.3f}".format(e, sum(mae[e])/len(mae[e]), sum(mse[e])/len(mse[e]))
        
            print("prior_mu_mean:",np.mean(prior_mean))
            print("phi_mu_mean: ",np.mean(phi_mean))

        writer.writerows(logs)

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
	save_path = args['save_path']#+'/aggVSdisag_distrib/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
	period = int(args['period'])
	n_steps = int(args['n_steps'])
	stride_train = int(args['stride_train'])
	stride_test = n_steps
	typeLoad = int(args['typeLoad'])

	flgMSE = int(args['flgMSE'])
	monitoring_freq = int(args['monitoring_freq'])
	epoch = int(args['epoch'])
	batch_size = int(args['batch_size'])
	x_dim = int(args['x_dim'])
	y_dim = int(args['y_dim'])
	z_dim = int(args['z_dim'])
	rnn_dim = int(args['rnn_dim'])
	k = int(args['num_k']) #a mixture of K Gaussian functions
	lr = float(args['lr'])
	origLR = lr
	debug = int(args['debug'])

	print "-------------------------------------------------------------------\n"
	print "trial no. %d" % trial
	print "learning rate %f" % lr
	print "saving pkl file '%s'" % pkl_name
	print "to the save path '%s'" % save_path
	print "Epochs: %d" % epoch
	print "-------------------------------------------------------------------\n"

	n_appl = 5 # number of appliance or the size of 3rd dimension

	#Data Preprocessing

	Xtrain, ytrain, Xval, yval, Xtest, ytest, reader = fetch_ukdale(data_path, windows, appliances,numApps=-1, period=period,n_steps= n_steps, 
                                              stride_train = stride_train, stride_test = stride_test,
                                              flgAggSumScaled = 1, flgFilterZeros = 1, typeLoad = typeLoad)

	print(np.shape(Xtrain))
	#file name to save test data
	save_file = save_path+'/testData_'+str(epoch)+'_'+str(lr)+".npy"

	if not os.path.exists(save_file):
		np.save(save_file,np.array([np.reshape(Xtest,[-1,]),np.reshape(ytest,[-1,])]))

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
	parser.add_argument('--test_file', type=str, default=save_file,
                        help='test data')
	args = parser.parse_args()

	model = VRNN(args) #create the VRNN model

	states = train(args, model,(Xtrain,ytrain), (Xval, yval))
	
	#save the preprocessed test data
	

if __name__ == "__main__":

    import sys, time
    if len(sys.argv) > 1:
        config_file_name = sys.argv[-1]
    else:
        config_file_name = 'config_AE-all.txt'

    f = open(config_file_name, 'r')
    lines = f.readlines()
    params = OrderedDict()

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')
        param_name = param_list[0]
        param_value = param_list[1]
        params[param_name] = param_value

    params['save_path'] = params['save_path']+'/allAtOnce/'+datetime.now().strftime("%y-%m-%d_%H-%M")
    os.makedirs(params['save_path'])
    shutil.copy('config_AE-all.txt', params['save_path']+'/config_AE-all.txt')

    main(params)
