import numpy as np
import scipy.io
import theano
import theano.typed_list as TL
import theano.tensor as T
import datetime
import shutil
import os
import matplotlib.pyplot as plt
plt.switch_backend('PDF')
import pickle
from cle.cle.cost import BiGMM, KLGaussianGaussian, GMMdisagMulti
from cle.cle.data import Iterator
from cle.cle.models import Model
from cle.cle.layers import InitCell
from cle.cle.layers.feedforward import FullyConnectedLayer
from cle.cle.layers.recurrent import LSTM
from cle.cle.train import Training
from cle.cle.train.ext import (
    EpochCount,
    GradientClipping,
    Monitoring,
    Picklize,
    EarlyStopping,
    WeightNorm
)
from cle.cle.train.opt import Adam
from cle.cle.utils import init_tparams, sharedX
from cle.cle.utils.compat import OrderedDict
from cle.cle.utils.op import Gaussian_sample, GMM_sample, GMM_sampleY
from cle.cle.utils.gpu_op import concatenate

from preprocessing.redd import Redd
from preprocessing.redd_utils import fetch_redd

appliances = ["fridge","dish washer", "light", "microwave"]#"microwave"
#[ 'air1', 'furnace1','refrigerator1', 'clotheswasher1','drye1','dishwasher1', 'kitchenapp1','microwave1']
windows = {3: ("2011-04-18", "2011-05-25")}#3413:("2015-06-01", "2015-12-31")
#windows = {6990:("2015-06-01", "2015-11-01"), 2859:("2015-06-01", "2015-11-01"), 7951:("2015-06-01", "2015-11-01"),8292:("2015-06-01",  "2015-11-01"),3413:("2015-06-01", "2015-11-01")}#3413:("2015-06-01", "2015-12-31")

def main(args):
    
    theano.optimizer='fast_compile'
    #theano.config.exception_verbosity='high'
    

    trial = int(args['trial'])
    pkl_name = 'dp_disall-sch_%d' % trial
    channel_name = 'mae'

    data_path = args['data_path']
    save_path = args['save_path']#+'/aggVSdisag_distrib/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    pickleModel = args['pickleModel']
    period = int(args['period'])
    n_steps = int(args['n_steps'])
    stride_train = int(args['stride_train'])
    stride_test = int(args['stride_test'])
    loadType = int(args['loadType'])

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
    kSchedSamp = int(args['kSchedSamp'])

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path
    print(str(windows))

    q_z_dim = 500
    p_z_dim = 500
    p_x_dim = 500
    x2s_dim = 200
    y2s_dim = 200
    z2s_dim = 200
    target_dim = k# As different appliances are separeted in theta_mu1, theta_mu2, etc... each one is just created from k different Gaussians

    model = Model()
    Xtrain, ytrain, Xval, yval, Xtest,ytest, reader = fetch_redd(data_path, windows, appliances,numApps=-1, period=period,
                                              n_steps= n_steps, stride_train = stride_train, stride_test = stride_test,
                                              trainPer=0.5, valPer=0.25, testPer=0.25, typeLoad = loadType,
                                              flgAggSumScaled = 1, flgFilterZeros = 1)

    print("Mean ",reader.meanTraining)
    print("Std", reader.stdTraining)
    instancesPlot = {0:[4]}

    train_data = Redd(name='train',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         inputX=Xtrain,
                         labels=ytrain)

    X_mean = train_data.X_mean
    X_std = train_data.X_std

    valid_data = Redd(name='valid',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         X_mean=X_mean,
                         X_std=X_std,
                         inputX=Xval,
                         labels = yval)

    test_data = Redd(name='valid',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         X_mean=X_mean,
                         X_std=X_std,
                         inputX=Xtest,
                         labels = ytest)

    init_W = InitCell('rand')
    init_U = InitCell('ortho')
    init_b = InitCell('zeros')
    init_b_sig = InitCell('const', mean=0.6)

    x, mask, y , y_mask = train_data.theano_vars()
    scheduleSamplingMask = T.fvector('schedMask')

    x.name = 'x_original'

    if debug:
        x.tag.test_value = np.zeros((15, batch_size, x_dim), dtype=np.float32)
        temp = np.ones((15, batch_size), dtype=np.float32)
        temp[:, -2:] = 0.
        mask.tag.test_value = temp

    #from experiment 18-05-31_18-48
    fmodel = open(pickleModel, 'rb')
    mainloop = pickle.load(fmodel)
    fmodel.close()

    #define layers
    rnn = mainloop.model.nodes[0]
    x_1 = mainloop.model.nodes[1]
    y_1 = mainloop.model.nodes[2]
    z_1 = mainloop.model.nodes[3]
    phi_1 = mainloop.model.nodes[4]
    phi_mu = mainloop.model.nodes[5]
    phi_sig = mainloop.model.nodes[6]
    prior_1 = mainloop.model.nodes[7]
    prior_mu = mainloop.model.nodes[8]
    prior_sig = mainloop.model.nodes[9]
    theta_1 = mainloop.model.nodes[10]
    theta_mu1 = mainloop.model.nodes[11]
    theta_sig1 = mainloop.model.nodes[12]
    coeff1 = mainloop.model.nodes[13]

    nodes = [rnn,
             x_1, y_1,z_1, #dissag_pred,
             phi_1, phi_mu, phi_sig,
             prior_1, prior_mu, prior_sig,
             theta_1, theta_mu1, theta_sig1, coeff1]

    params = mainloop.model.params

    dynamicOutput = [None, None, None, None, None, None, None, None]
    #dynamicOutput_val = [None, None, None, None, None, None,None,  None, None]
    if (y_dim>1):
      theta_mu2 = mainloop.model.nodes[14]
      theta_sig2 = mainloop.model.nodes[15]
      coeff2 = mainloop.model.nodes[16]
      nodes = nodes + [theta_mu2, theta_sig2, coeff2]
      dynamicOutput = dynamicOutput+[None, None, None, None] #mu, sig, coef and pred
    if (y_dim>2):
      theta_mu3 = mainloop.model.nodes[17]
      theta_sig3 = mainloop.model.nodes[18]
      coeff3 = mainloop.model.nodes[19]
      nodes = nodes + [theta_mu3, theta_sig3, coeff3]
      dynamicOutput = dynamicOutput +[None, None, None, None]
    if (y_dim>3):
      theta_mu4 = mainloop.model.nodes[20]
      theta_sig4 = mainloop.model.nodes[21]
      coeff4 = mainloop.model.nodes[22]
      nodes = nodes + [theta_mu4, theta_sig4, coeff4]
      dynamicOutput = dynamicOutput + [None, None, None, None]


    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)
    y_1_temp = y_1.fprop([y], params)

    output_fn = [s_0] + dynamicOutput
    output_fn_val = [s_0] + dynamicOutput[2:]
    print(len(output_fn), len(output_fn_val))


    def inner_fn_test(x_t, s_tm1):

        prior_1_t = prior_1.fprop([x_t,s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(prior_mu_t, prior_sig_t)#in the original code it is gaussian. GMM is for the generation
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)
        theta_mu1_t = theta_mu1.fprop([theta_1_t], params)
        theta_sig1_t = theta_sig1.fprop([theta_1_t], params)
        coeff1_t = coeff1.fprop([theta_1_t], params)

        y_pred1 = GMM_sampleY(theta_mu1_t, theta_sig1_t, coeff1_t) #Gaussian_sample(theta_mu_t, theta_sig_t)

        tupleMulti = prior_mu_t, prior_sig_t, theta_mu1_t, theta_sig1_t, coeff1_t, y_pred1

        if (y_dim>1):
          theta_mu2_t = theta_mu2.fprop([theta_1_t], params)
          theta_sig2_t = theta_sig2.fprop([theta_1_t], params)
          coeff2_t = coeff2.fprop([theta_1_t], params)
          y_pred2 = GMM_sampleY(theta_mu2_t, theta_sig2_t, coeff2_t)
          y_pred1 = T.concatenate([y_pred1, y_pred2],axis=1)
          tupleMulti = tupleMulti + (theta_mu2_t, theta_sig2_t, coeff2_t, y_pred2)

        if (y_dim>2):
          theta_mu3_t = theta_mu3.fprop([theta_1_t], params)
          theta_sig3_t = theta_sig3.fprop([theta_1_t], params)
          coeff3_t = coeff3.fprop([theta_1_t], params)
          y_pred3 = GMM_sampleY(theta_mu3_t, theta_sig3_t, coeff3_t)
          y_pred1 = T.concatenate([y_pred1, y_pred3],axis=1)
          tupleMulti = tupleMulti + (theta_mu3_t, theta_sig3_t, coeff3_t, y_pred3)

        if (y_dim>3):
          theta_mu4_t = theta_mu4.fprop([theta_1_t], params)
          theta_sig4_t = theta_sig4.fprop([theta_1_t], params)
          coeff4_t = coeff4.fprop([theta_1_t], params)
          y_pred4 = GMM_sampleY(theta_mu4_t, theta_sig4_t, coeff4_t)
          y_pred1 = T.concatenate([y_pred1, y_pred4],axis=1)
          tupleMulti = tupleMulti + (theta_mu4_t, theta_sig4_t, coeff4_t, y_pred4)

        pred_1_t=y_1.fprop([y_pred1], params)
        #y_pred = [GMM_sampleY(theta_mu_t[i], theta_sig_t[i], coeff_t[i]) for i in range(y_dim)]#T.stack([y_pred1,y_pred2],axis = 0 )
        s_t = rnn.fprop([[x_t, z_1_t, pred_1_t], [s_tm1]], params)
        #y_pred = dissag_pred.fprop([s_t], params)

        return (s_t,)+tupleMulti
        #corr_temp, binary_temp
    (otherResults_val, updates_val) = theano.scan(fn=inner_fn_test, sequences=[x_1_temp],
                            outputs_info=output_fn_val )

    for k, v in updates_val.iteritems():
        k.default_update = v


    x_shape = x.shape
    y_shape = y.shape
    x_in = x.reshape((x_shape[0]*x_shape[1], -1))
    y_in = y.reshape((y_shape[0]*y_shape[1], -1))


    ######################## TEST (GENERATION) TIME
    s_temp_val, prior_mu_temp_val, prior_sig_temp_val, \
      theta_mu1_temp_val, theta_sig1_temp_val, coeff1_temp_val, y_pred1_temp_val = otherResults_val[:7]
    restResults_val = otherResults_val[7:]

    #s_temp_val = concatenate([s_0[None, :, :], s_temp_val[:-1]], axis=0)# seems like this is for creating an additional dimension to s_0

    theta_mu1_temp_val.name = 'theta_mu1_val'
    theta_sig1_temp_val.name = 'theta_sig1_val'
    coeff1_temp_val.name = 'coeff1_val'
    y_pred1_temp_val.name = 'disaggregation1_val'
    y_pred1_temp_val = T.clip(y_pred1_temp_val,0.0,np.inf)
    prediction_val = y_pred1_temp_val

    #[:,:,flgAgg].reshape((y.shape[0],y.shape[1],1)
    mse1_val = T.mean((y_pred1_temp_val - y[:,:,0].reshape((y.shape[0],y.shape[1],1)))**2)
    mae1_val = T.mean( T.abs_(y_pred1_temp_val - y[:,:,0].reshape((y.shape[0],y.shape[1],1))) )

    totPred = T.sum(y_pred1_temp_val)
    totReal = T.sum(y[:,:,0])
    relErr1_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
    propAssigned1_val = 1 - T.sum(T.abs_(y_pred1_temp_val - y[:,:,0].reshape((y.shape[0],y.shape[1],1))))/(2*T.sum(x))

    #y_unNormalize = (y[:,:,0] * reader.stdTrain[0]) + reader.meanTrain[0]
    #y_pred1_temp_val = (y_pred1_temp_val * reader.stdTrain[0]) + reader.meanTrain[0]
    #mse1_valUnNorm = T.mean((y_pred1_temp_val - y_unNormalize.reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
    #mae1_valUnNorm = T.mean( T.abs_(y_pred1_temp_val - y_unNormalize.reshape((y.shape[0],y.shape[1],1))) )

    mse1_val.name = 'mse1_val'
    mae1_val.name = 'mae1_val'

    theta_mu1_in_val = theta_mu1_temp_val.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig1_in_val = theta_sig1_temp_val.reshape((x_shape[0]*x_shape[1], -1))
    coeff1_in_val = coeff1_temp_val.reshape((x_shape[0]*x_shape[1], -1))

    totaMSE_val = mse1_val
    totaMAE_val =mae1_val
    indexSepDynamic_val = 5

    #Initializing values of mse and mae
    mse2_val = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae2_val = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mse3_val = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae3_val = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mse4_val = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae4_val = T.mean(T.zeros((y.shape[0],y.shape[1],1)))

    '''
    mse2_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mse3_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mse4_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mse5_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mse6_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mse7_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mse8_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae2_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae3_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae4_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae5_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae6_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae7_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    mae8_valUnNorm = T.mean(T.zeros((y.shape[0],y.shape[1],1)))
    '''

    relErr2_val = T.zeros((1,))
    relErr3_val = T.zeros((1,))
    relErr4_val = T.zeros((1,))

    propAssigned2_val = T.zeros((1,))
    propAssigned3_val = T.zeros((1,))
    propAssigned4_val = T.zeros((1,))

    if (y_dim>1):
      theta_mu2_temp_val, theta_sig2_temp_val, coeff2_temp_val, y_pred2_temp_val = restResults_val[:4]
      restResults_val = restResults_val[4:]
      theta_mu2_temp_val.name = 'theta_mu2_val'
      theta_sig2_temp_val.name = 'theta_sig2_val'
      coeff2_temp_val.name = 'coeff2_val'
      y_pred2_temp_val.name = 'disaggregation2_val'
      y_pred2_temp_val = T.clip(y_pred2_temp_val,0.0,np.inf)
      prediction_val = T.concatenate([prediction_val, y_pred2_temp_val], axis=2) #before it gets unnormalized

      mse2_val = T.mean((y_pred2_temp_val - y[:,:,1].reshape((y.shape[0],y.shape[1],1)))**2)
      mae2_val = T.mean( T.abs_(y_pred2_temp_val - y[:,:,1].reshape((y.shape[0],y.shape[1],1))) )

      totPred = T.sum(y_pred2_temp_val)
      totReal = T.sum(y[:,:,1])
      relErr2_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
      propAssigned2_val = 1 - T.sum(T.abs_(y_pred2_temp_val - y[:,:,1].reshape((y.shape[0],y.shape[1],1))))/(2*T.sum(x))

      #y_unNormalize = (y[:,:,1] * reader.stdTrain[1]) + reader.meanTrain[1]
      #y_pred2_temp_val = (y_pred2_temp_val * reader.stdTrain[1]) + reader.meanTrain[1]
      #mse2_valUnNorm = T.mean((y_pred2_temp_val - y_unNormalize.reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      #mae2_valUnNorm = T.mean( T.abs_(y_pred2_temp_val - y_unNormalize.reshape((y.shape[0],y.shape[1],1))) )

      mse2_val.name = 'mse2_val'
      mae2_val.name = 'mae2_val'

      theta_mu2_in_val = theta_mu2_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig2_in_val = theta_sig2_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      coeff2_in_val = coeff2_temp_val.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM_val = theta_mu2_in_val, theta_sig2_in_val, coeff2_in_val

      totaMSE_val+=mse2_val
      totaMAE_val+=mae2_val
      indexSepDynamic_val +=2

    if (y_dim>2):
      theta_mu3_temp_val, theta_sig3_temp_val, coeff3_temp_val, y_pred3_temp_val = restResults_val[:4]
      restResults_val = restResults_val[4:]
      theta_mu3_temp_val.name = 'theta_mu3_val'
      theta_sig3_temp_val.name = 'theta_sig3_val'
      coeff3_temp_val.name = 'coeff3_val'
      y_pred3_temp_val.name = 'disaggregation3_val'
      y_pred3_temp_val = T.clip(y_pred3_temp_val,0.0,np.inf)
      prediction_val = T.concatenate([prediction_val, y_pred3_temp_val], axis=2) #before it gets unnormalized

      mse3_val = T.mean((y_pred3_temp_val - y[:,:,2].reshape((y.shape[0],y.shape[1],1)))**2)
      mae3_val = T.mean( T.abs_(y_pred3_temp_val - y[:,:,2].reshape((y.shape[0],y.shape[1],1))) )

      totPred = T.sum(y_pred3_temp_val)
      totReal = T.sum(y[:,:,2])
      relErr3_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
      propAssigned3_val = 1 - T.sum(T.abs_(y_pred3_temp_val - y[:,:,2].reshape((y.shape[0],y.shape[1],1))))/(2*T.sum(x))

      #y_unNormalize = (y[:,:,2] * reader.stdTrain[2]) + reader.meanTrain[2]
      #y_pred3_temp_val = (y_pred3_temp_val * reader.stdTrain[2]) + reader.meanTrain[2]
      #mse3_valUnNorm = T.mean((y_pred3_temp_val - y_unNormalize.reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      #mae3_valUnNorm = T.mean( T.abs_(y_pred3_temp_val - y_unNormalize.reshape((y.shape[0],y.shape[1],1))) )

      mse3_val.name = 'mse3_val'
      mae3_val.name = 'mae3_val'

      theta_mu3_in_val = theta_mu3_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig3_in_val = theta_sig3_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      coeff3_in_val = coeff3_temp_val.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM_val = argsGMM_val + (theta_mu3_in_val, theta_sig3_in_val, coeff3_in_val)
      totaMSE_val+=mse3_val
      totaMAE_val+=mae3_val
      indexSepDynamic_val +=2

      

    if (y_dim>3):
      theta_mu4_temp_val, theta_sig4_temp_val, coeff4_temp_val, y_pred4_temp_val = restResults_val[:4]
      restResults_val = restResults_val[4:]
      theta_mu4_temp_val.name = 'theta_mu4_val'
      theta_sig4_temp_val.name = 'theta_sig4_val'
      coeff4_temp_val.name = 'coeff4_val'
      y_pred4_temp_val.name = 'disaggregation4_val'
      y_pred4_temp_val = T.clip(y_pred4_temp_val,0.0,np.inf)

      prediction_val = T.concatenate([prediction_val, y_pred4_temp_val], axis=2) #before it gets unnormalized

      mse4_val = T.mean((y_pred4_temp_val - y[:,:,3].reshape((y.shape[0],y.shape[1],1)))**2)
      mae4_val = T.mean( T.abs_(y_pred4_temp_val - y[:,:,3].reshape((y.shape[0],y.shape[1],1))) )

      totPred = T.sum(y_pred4_temp_val)
      totReal = T.sum(y[:,:,3])
      relErr4_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
      propAssigned4_val = 1 - T.sum(T.abs_(y_pred4_temp_val - y[:,:,3].reshape((y.shape[0],y.shape[1],1))))/(2*T.sum(x))

      mse4_val.name = 'mse4_val'
      mae4_val.name = 'mae4_val'

      theta_mu4_in_val = theta_mu4_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig4_in_val = theta_sig4_temp_val.reshape((x_shape[0]*x_shape[1], -1))
      coeff4_in_val = coeff4_temp_val.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM_val = argsGMM_val + (theta_mu4_in_val, theta_sig4_in_val, coeff4_in_val)
      totaMSE_val+=mse4_val
      totaMAE_val+=mae4_val
      indexSepDynamic_val +=2

    recon_val = GMMdisagMulti(y_dim, y_in, theta_mu1_in_val, theta_sig1_in_val, coeff1_in_val, *argsGMM_val)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon_val = recon_val.reshape((x_shape[0], x_shape[1]))
    recon_val.name = 'gmm_out'
    totaMSE_val = totaMSE_val/y_dim
    totaMAE_val = totaMAE_val/y_dim

    recon_term_val = recon_val.sum(axis=0).mean()
    recon_term_val = recon_val.sum(axis=0).mean()
    recon_term_val.name = 'recon_term'

    ######################

    data=Iterator(test_data, batch_size)

    test_fn = theano.function(inputs=[x, y],#[x, y],
                              #givens={x:Xtest},
                              #on_unused_input='ignore',
                              #z=( ,200,1)
                              allow_input_downcast=True,
                              outputs=[prediction_val, recon_term_val, totaMSE_val, totaMAE_val, 
                                        mse1_val,mse2_val,mse3_val,mse4_val,
                                        mae1_val,mae2_val,mae3_val,mae4_val, #unnormalized mae and mse 16 items#
                                        relErr1_val,relErr2_val,relErr3_val,relErr4_val,
                                        propAssigned1_val, propAssigned2_val,propAssigned3_val,propAssigned4_val],
                              updates=updates_val
                              )
    testOutput = []
    testMetrics2 = []
    perEnergyAssig = []

    bestInstsancesPred = []
    bestInstsancesDisa = []
    bestInstsancesAggr = []

    numBatchTest = 0
    for batch in data:
      outputGeneration = test_fn(batch[0], batch[2])
      testOutput.append(outputGeneration[1:12]) #before 36 including unnormalized metrics
      testMetrics2.append(outputGeneration[12:])
      #{0:[4,20], 2:[5,10]} 
      #if (numBatchTest==0):

      ########## best mae
      predTest = np.transpose(outputGeneration[0],[1,0,2]).clip(min=0)
      realTest = np.transpose(batch[2],[1,0,2])

      batchMSE = np.mean(np.absolute(predTest-realTest),axis=(1,2))
      idxMin = np.argmin(batchMSE)

      #print(np.asarray(idxMin).reshape(1,-1)[0,:])
      #print(batchMSE[idxMin])
      for idx in np.asarray(idxMin).reshape(1,-1)[0,:]:

        plt.figure(1)
        plt.plot(predTest[idx])
        plt.legend(appliances)
        plt.savefig(save_path+"/vrnn_disall_test-b{}_Pred_0-{}".format(numBatchTest,idx))
        plt.clf()

        plt.figure(2)
        plt.plot(realTest[idx])
        plt.legend(appliances)
        plt.savefig(save_path+"/vrnn_disall_test-b{}_RealDisag_0-{}".format(numBatchTest,idx))
        plt.clf()

        plt.figure(3)
        plt.plot(np.transpose(batch[0],[1,0,2])[idx])
        plt.savefig(save_path+"/vrnn_disall_test-b{}_Realagg_0-{}".format(numBatchTest,idx))
        plt.clf()

        bestInstsancesPred.append(predTest[idx])
        bestInstsancesDisa.append(realTest[idx])
        bestInstsancesAggr.append(np.transpose(batch[0],[1,0,2])[idx])

      numBatchTest+=1

      sumNumPred = np.sum(predTest, axis=(0,1))
      sumNumReal = np.sum(batch[2], axis=(0,1))
      perEnergy  = np.sum(batch[0], axis=(0,1))
      perEnergyAssig.append((sumNumReal/perEnergy,sumNumPred/perEnergy))

    scipy.io.savemat(save_path+'/testInstances.mat', mdict={'pred': bestInstsancesPred, 'disag':bestInstsancesDisa, 'agg':bestInstsancesAggr})

    testOutput = np.asarray(testOutput)
    testMetrics2 = np.asarray(testMetrics2)
    print(testOutput.shape)
    print(testMetrics2.shape)

    #testOutput[:,19:] = 1000 * testOutput[:,19:] # kwtts a watts
    recon_test = testOutput[:, 0].mean()
    mse_test =  testOutput[:, 1].mean()
    mae_test =  testOutput[:, 2].mean()
    mse1_test =  testOutput[:, 3].mean()
    mae1_test =  testOutput[:, 7].mean()
    mse2_test =  testOutput[:, 4].mean()
    mae2_test =  testOutput[:, 8].mean()
    mse3_test =  testOutput[:, 5].mean()
    mae3_test =  testOutput[:, 9].mean()
    mse4_test =  testOutput[:, 6].mean()
    mae4_test =  testOutput[:, 10].mean()


    relErr1_test = testMetrics2[:,0].mean()
    relErr2_test = testMetrics2[:,1].mean()
    relErr3_test = testMetrics2[:,2].mean()
    relErr4_test = testMetrics2[:,3].mean()

    propAssigned1_test = testMetrics2[:, 4].mean()
    propAssigned2_test = testMetrics2[:, 5].mean()
    propAssigned3_test = testMetrics2[:, 6].mean()
    propAssigned4_test = testMetrics2[:, 7].mean()

    fLog = open(save_path+'/output.csv', 'w')

    #fLog.write(str(lr_iterations)+"\n")
    fLog.write(pickleModel+"\n")
    fLog.write(str(appliances)+"\n")
    fLog.write(str(windows)+"\n\n")
    fLog.write("logTest,mse1_test,mse2_test,mse3_test,mse4_test,mae1_test,mae2_test,mae3_test,mae4_test,mseTest,maeTest\n")
    #fLog.write("Unnorm,{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},0.0,0.0\n\n".format(mse1_valUnNorm,mse2_valUnNorm,mse3_valUnNorm,mse4_valUnNorm,mse5_valUnNorm, mse6_valUnNorm,mse7_valUnNorm,mse8_valUnNorm,mae1_valUnNorm,mae2_valUnNorm,mae3_valUnNorm,mae4_valUnNorm,mae5_valUnNorm, mae6_valUnNorm,mae7_valUnNorm,mae8_valUnNorm))
    fLog.write("{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n\n".format(recon_test,mse1_test,mse2_test,mse3_test,mse4_test,mae1_test,mae2_test,mae3_test,mae4_test,mse_test,mae_test))
    fLog.write("relErr1,relErr2,relErr3,relErr4,propAssigned1,propAssigned2,propAssigned3,propAssigned4\n")
    fLog.write("{},{},{},{},{},{},{},{}\n".format(relErr1_test,relErr2_test,relErr3_test,relErr4_test,propAssigned1_test,propAssigned2_test,propAssigned3_test, propAssigned4_test))

    fLog.write("batch,perReal1,perReal2,perReal3,perReal4,perPredict1,perPredict2,perPredict3,perPredict4\n")
    for batch, item in enumerate(perEnergyAssig):
      fLog.write("{},{},{},{},{},{},{},{},{}\n".format(batch,item[0][0],item[0][1],item[0][2],item[0][3],item[1][0],item[1][1],item[1][2],item[1][3]))

    '''
    fLog.write("q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim\n")
    fLog.write("{},{},{},{},{},{}\n".format(q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim))
    fLog.write("epoch,log,kl,mse1,mse2,mse3,mse4,mae1,mae2,mae3,mae4\n")
    for i , item in enumerate(mainloop.trainlog.monitor['nll_upper_bound']):
      e,f,g,h,j,k,l,n,p,q,r,s,t,u =  0,0,0,0,0,0,0,0,0,0,0,0,0,0
      ep = mainloop.trainlog.monitor['epoch'][i]
      a = mainloop.trainlog.monitor['recon_term'][i]
      b = mainloop.trainlog.monitor['kl_term'][i]
      d = mainloop.trainlog.monitor['mse1'][i]
      m = mainloop.trainlog.monitor['mae1'][i]
      
      if (y_dim>1):
        e = mainloop.trainlog.monitor['mse2'][i]
        n = mainloop.trainlog.monitor['mae2'][i]
      if (y_dim>2):
        f = mainloop.trainlog.monitor['mse3'][i]
        p = mainloop.trainlog.monitor['mae3'][i]
      if (y_dim>3):
        g = mainloop.trainlog.monitor['mse4'][i]
        q = mainloop.trainlog.monitor['mae4'][i]

      fLog.write("{:d},{:.2f},{:.2f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
                  ep,a,b,d,e,f,g,m,n,p,q))
    f = open(save_path+'/outputRealGeneration.pkl', 'wb')
    pickle.dump(outputGeneration, f, -1)
    f.close()
    '''

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

    params['save_path'] = params['save_path']+'/allAtOnce/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    os.makedirs(params['save_path'])
    shutil.copy('config_AE-all.txt', params['save_path']+'/config_AE-all.txt')

    main(params)