import numpy as np
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

appliances = ["fridge","dish washer", "light","microwave"] #if testing in 6, not using "microwave"
# FOR HOUSE 6, OMIT DISH WASHER
windows = {3: ("2011-04-01", "2011-05-30")} 
def main(args):
    
    theano.optimizer='fast_compile'
    #theano.config.exception_verbosity='high'
    

    trial = int(args['trial'])
    pkl_name = 'dp_disall-sch_%d' % trial
    channel_name = 'mae'

    data_path = args['data_path']
    save_path = args['save_path']#+'/aggVSdisag_distrib/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
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
    typeActivFunc = args['typeActivFunc']

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
    lr_iterations = {0:lr}
    
    target_dim = k# As different appliances are separeted in theta_mu1, theta_mu2, etc... each one is just created from k different Gaussians

    model = Model()
    Xtrain, ytrain, Xval, yval, Xtest,ytest, reader = fetch_redd(data_path, windows, appliances,numApps=-1, period=period,
                                              n_steps= n_steps, stride_train = stride_train, stride_test = stride_test,
                                              trainPer=0.5, valPer=0.25, testPer=0.25, typeLoad = loadType,
                                              flgAggSumScaled = 1, flgFilterZeros = 1)

    print(Xtrain.shape, Xval.shape, Xtest.shape, ytrain.shape, yval.shape, ytest.shape)
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

    x_1 = FullyConnectedLayer(name='x_1',
                              parent=['x_t'],
                              parent_dim=[x_dim],
                              nout=x2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    y_1 = FullyConnectedLayer(name='y_1',
                              parent=['y_t'],
                              parent_dim=[y_dim],
                              nout=y2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    z_1 = FullyConnectedLayer(name='z_1',
                              parent=['z_t'],
                              parent_dim=[z_dim],
                              nout=z2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    rnn = LSTM(name='rnn',
               parent=['x_1', 'z_1', 'y_1'],
               parent_dim=[x2s_dim, z2s_dim, y2s_dim],
               nout=rnn_dim,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

    phi_1 = FullyConnectedLayer(name='phi_1',
                                parent=['x_1', 's_tm1','y_1'],
                                parent_dim=[x2s_dim, rnn_dim, y2s_dim],
                                nout=q_z_dim,
                                unit='relu',
                                init_W=init_W,
                                init_b=init_b)

    phi_mu = FullyConnectedLayer(name='phi_mu',
                                 parent=['phi_1'],
                                 parent_dim=[q_z_dim],
                                 nout=z_dim,
                                 unit='linear',
                                 init_W=init_W,
                                 init_b=init_b)

    phi_sig = FullyConnectedLayer(name='phi_sig',
                                  parent=['phi_1'],
                                  parent_dim=[q_z_dim],
                                  nout=z_dim,
                                  unit='softplus',
                                  cons=1e-4,
                                  init_W=init_W,
                                  init_b=init_b_sig)

    prior_1 = FullyConnectedLayer(name='prior_1',
                                  parent=['x_1','s_tm1'],
                                  parent_dim=[x2s_dim,rnn_dim],
                                  nout=p_z_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    prior_mu = FullyConnectedLayer(name='prior_mu',
                                   parent=['prior_1'],
                                   parent_dim=[p_z_dim],
                                   nout=z_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    prior_sig = FullyConnectedLayer(name='prior_sig',
                                    parent=['prior_1'],
                                    parent_dim=[p_z_dim],
                                    nout=z_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    theta_1 = FullyConnectedLayer(name='theta_1',
                                  parent=['z_1', 's_tm1'],
                                  parent_dim=[z2s_dim, rnn_dim],
                                  nout=p_x_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    theta_mu1 = FullyConnectedLayer(name='theta_mu1',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit=typeActivFunc,
                                   init_W=init_W,
                                   init_b=init_b)

    if (y_dim>1):
      theta_mu2 = FullyConnectedLayer(name='theta_mu2',
                                     parent=['theta_1'],
                                     parent_dim=[p_x_dim],
                                     nout=target_dim,
                                     unit=typeActivFunc,
                                     init_W=init_W,
                                     init_b=init_b)

    if (y_dim>2):
      theta_mu3 = FullyConnectedLayer(name='theta_mu3',
                                     parent=['theta_1'],
                                     parent_dim=[p_x_dim],
                                     nout=target_dim,
                                     unit=typeActivFunc,
                                     init_W=init_W,
                                     init_b=init_b)

    if (y_dim>3):
      theta_mu4 = FullyConnectedLayer(name='theta_mu4',
                                     parent=['theta_1'],
                                     parent_dim=[p_x_dim],
                                     nout=target_dim,
                                     unit=typeActivFunc,
                                     init_W=init_W,
                                     init_b=init_b)


    theta_sig1 = FullyConnectedLayer(name='theta_sig1',
                                    parent=['theta_1'],
                                    parent_dim=[p_x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    if (y_dim>1):
      theta_sig2 = FullyConnectedLayer(name='theta_sig2',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=target_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

    if (y_dim>2):
      theta_sig3 = FullyConnectedLayer(name='theta_sig3',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=target_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

    if (y_dim>3):
      theta_sig4 = FullyConnectedLayer(name='theta_sig4',
                                      parent=['theta_1'],
                                      parent_dim=[p_x_dim],
                                      nout=target_dim,
                                      unit='softplus',
                                      cons=1e-4,
                                      init_W=init_W,
                                      init_b=init_b_sig)

    coeff1 = FullyConnectedLayer(name='coeff1',
                                parent=['theta_1'],
                                parent_dim=[p_x_dim],
                                nout=k,
                                unit='softmax',
                                init_W=init_W,
                                init_b=init_b)

    if (y_dim>1):
      coeff2 = FullyConnectedLayer(name='coeff2',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    if (y_dim>2):
      coeff3 = FullyConnectedLayer(name='coeff3',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    if (y_dim>3): 
      coeff4 = FullyConnectedLayer(name='coeff4',
                                  parent=['theta_1'],
                                  parent_dim=[p_x_dim],
                                  nout=k,
                                  unit='softmax',
                                  init_W=init_W,
                                  init_b=init_b)

    corr = FullyConnectedLayer(name='corr',
                               parent=['theta_1'],
                               parent_dim=[p_x_dim],
                               nout=k,
                               unit='tanh',
                               init_W=init_W,
                               init_b=init_b)

    binary = FullyConnectedLayer(name='binary',
                                 parent=['theta_1'],
                                 parent_dim=[p_x_dim],
                                 nout=1,
                                 unit='sigmoid',
                                 init_W=init_W,
                                 init_b=init_b)

    nodes = [rnn,
             x_1, y_1,z_1, #dissag_pred,
             phi_1, phi_mu, phi_sig,
             prior_1, prior_mu, prior_sig,
             theta_1, theta_mu1, theta_sig1, coeff1]

    dynamicOutput = [None, None, None, None, None, None, None, None]
    if (y_dim>1):
      nodes = nodes + [theta_mu2, theta_sig2, coeff2]
      dynamicOutput = dynamicOutput+[None, None, None, None] #mu, sig, coef and pred
    if (y_dim>2):
      nodes = nodes + [theta_mu3, theta_sig3, coeff3]
      dynamicOutput = dynamicOutput +[None, None, None, None]
    if (y_dim>3):
      nodes = nodes + [theta_mu4, theta_sig4, coeff4]
      dynamicOutput = dynamicOutput + [None, None, None, None]

    params = OrderedDict()

    for node in nodes:
        if node.initialize() is not None:
            params.update(node.initialize())

    params = init_tparams(params)

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)
    y_1_temp = y_1.fprop([y], params)

    output_fn = [s_0] + dynamicOutput
    output_fn_val = [s_0] + dynamicOutput[2:]
    print(len(output_fn), len(output_fn_val))


    def inner_fn(x_t, y_t, scheduleSamplingMask, s_tm1):

        phi_1_t = phi_1.fprop([x_t, s_tm1, y_t], params)
        phi_mu_t = phi_mu.fprop([phi_1_t], params)
        phi_sig_t = phi_sig.fprop([phi_1_t], params)

        prior_1_t = prior_1.fprop([x_t, s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(phi_mu_t, phi_sig_t)#in the original code it is gaussian. GMM is for the generation
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)

        
        theta_mu1_t = theta_mu1.fprop([theta_1_t], params)
        theta_sig1_t = theta_sig1.fprop([theta_1_t], params)
        coeff1_t = coeff1.fprop([theta_1_t], params)

        ## prediction 1
        y_pred = GMM_sampleY(theta_mu1_t, theta_sig1_t, coeff1_t) #Gaussian_sample(theta_mu_t, theta_sig_t)

        tupleMulti = phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, theta_mu1_t, theta_sig1_t, coeff1_t, y_pred

        if (y_dim>1):
          theta_mu2_t = theta_mu2.fprop([theta_1_t], params)
          theta_sig2_t = theta_sig2.fprop([theta_1_t], params)
          coeff2_t = coeff2.fprop([theta_1_t], params)
          y_pred2 = GMM_sampleY(theta_mu2_t, theta_sig2_t, coeff2_t)
          y_pred = T.concatenate([y_pred, y_pred2],axis=1)
          tupleMulti = tupleMulti + (theta_mu2_t, theta_sig2_t, coeff2_t, y_pred2)

        if (y_dim>2):
          theta_mu3_t = theta_mu3.fprop([theta_1_t], params)
          theta_sig3_t = theta_sig3.fprop([theta_1_t], params)
          coeff3_t = coeff3.fprop([theta_1_t], params)
          y_pred3 = GMM_sampleY(theta_mu3_t, theta_sig3_t, coeff3_t)
          y_pred = T.concatenate([y_pred, y_pred3],axis=1)
          tupleMulti = tupleMulti + (theta_mu3_t, theta_sig3_t, coeff3_t, y_pred3)

        if (y_dim>3):
          theta_mu4_t = theta_mu4.fprop([theta_1_t], params)
          theta_sig4_t = theta_sig4.fprop([theta_1_t], params)
          coeff4_t = coeff4.fprop([theta_1_t], params)
          y_pred4 = GMM_sampleY(theta_mu4_t, theta_sig4_t, coeff4_t)
          y_pred = T.concatenate([y_pred, y_pred4],axis=1)
          tupleMulti = tupleMulti + (theta_mu4_t, theta_sig4_t, coeff4_t, y_pred4)

        #s_t = rnn.fprop([[x_t, z_1_t, y_t], [s_tm1]], params)
        
        if (scheduleSamplingMask==1):
          s_t = rnn.fprop([[x_t, z_1_t, y_t], [s_tm1]], params)
        else:
          y_t_aux = y_1.fprop([y_pred], params)
          s_t = rnn.fprop([[x_t, z_1_t, y_t_aux], [s_tm1]], params)
        
        return (s_t,)+tupleMulti

        #corr_temp, binary_temp
    (otherResults, updates) = theano.scan(fn=inner_fn, sequences=[x_1_temp, y_1_temp,scheduleSamplingMask ],
                            outputs_info=output_fn )#[s_0, (None)]

    s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp,\
      theta_mu1_temp, theta_sig1_temp, coeff1_temp, y_pred1_temp = otherResults[:9]
    restResults = otherResults[9:]

    for k, v in updates.iteritems():
        k.default_update = v

    #s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)# seems like this is for creating an additional dimension to s_0

    theta_mu1_temp.name = 'theta_mu1'
    theta_sig1_temp.name = 'theta_sig1'
    coeff1_temp.name = 'coeff1'
    y_pred1_temp.name = 'disaggregation1'

    #[:,:,flgAgg].reshape((y.shape[0],y.shape[1],1)
    mse1 = T.mean((y_pred1_temp - y[:,:,0].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
    mae1 = T.mean( T.abs_(y_pred1_temp - y[:,:,0].reshape((y.shape[0],y.shape[1],1))) )
    mse1.name = 'mse1'
    mae1.name = 'mae1'

    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

    x_shape = x.shape
    y_shape = y.shape
    x_in = x.reshape((x_shape[0]*x_shape[1], -1))
    y_in = y.reshape((y_shape[0]*y_shape[1], -1))

    theta_mu1_in = theta_mu1_temp.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig1_in = theta_sig1_temp.reshape((x_shape[0]*x_shape[1], -1))
    coeff1_in = coeff1_temp.reshape((x_shape[0]*x_shape[1], -1))


    ddoutMSEA = []
    ddoutYpreds = [y_pred1_temp]
    indexSepDynamic = 7 #plus two totalmse, totalmae

    totaMAE = T.copy(mae1)
    totaMSE = T.copy(mse1)
    mse2 = T.zeros((1,))
    mae2 = T.zeros((1,))
    mse3 = T.zeros((1,))
    mae3 = T.zeros((1,))
    mse4 = T.zeros((1,))
    mae4 = T.zeros((1,))

    if (y_dim>1):
      theta_mu2_temp, theta_sig2_temp, coeff2_temp, y_pred2_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu2_temp.name = 'theta_mu2'
      theta_sig2_temp.name = 'theta_sig2'
      coeff2_temp.name = 'coeff2'
      y_pred2_temp.name = 'disaggregation2'
      mse2 = T.mean((y_pred2_temp - y[:,:,1].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae2 = T.mean( T.abs_(y_pred2_temp - y[:,:,1].reshape((y.shape[0],y.shape[1],1))) )
      mse2.name = 'mse2'
      mae2.name = 'mae2'

      theta_mu2_in = theta_mu2_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig2_in = theta_sig2_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff2_in = coeff2_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = theta_mu2_in, theta_sig2_in, coeff2_in

      ddoutMSEA = ddoutMSEA + [mse2, mae2]
      ddoutYpreds = ddoutYpreds + [y_pred2_temp]
      #totaMSE+=mse2
      indexSepDynamic +=2

    if (y_dim>2):
      theta_mu3_temp, theta_sig3_temp, coeff3_temp, y_pred3_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu3_temp.name = 'theta_mu3'
      theta_sig3_temp.name = 'theta_sig3'
      coeff3_temp.name = 'coeff3'
      y_pred3_temp.name = 'disaggregation3'
      mse3 = T.mean((y_pred3_temp - y[:,:,2].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae3 = T.mean( T.abs_(y_pred3_temp - y[:,:,2].reshape((y.shape[0],y.shape[1],1))) )
      mse3.name = 'mse3'
      mae3.name = 'mae3'

      theta_mu3_in = theta_mu3_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig3_in = theta_sig3_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff3_in = coeff3_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = argsGMM + (theta_mu3_in, theta_sig3_in, coeff3_in)
      ddoutMSEA = ddoutMSEA + [mse3, mae3]
      ddoutYpreds = ddoutYpreds + [y_pred3_temp]
      #totaMSE+=mse3
      indexSepDynamic +=2

    if (y_dim>3):
      theta_mu4_temp, theta_sig4_temp, coeff4_temp, y_pred4_temp = restResults[:4]
      restResults = restResults[4:]
      theta_mu4_temp.name = 'theta_mu4'
      theta_sig4_temp.name = 'theta_sig4'
      coeff4_temp.name = 'coeff4'
      y_pred4_temp.name = 'disaggregation4'
      mse4 = T.mean((y_pred4_temp - y[:,:,3].reshape((y.shape[0],y.shape[1],1)))**2) # As axis = None is calculated for all
      mae4 = T.mean( T.abs_(y_pred4_temp - y[:,:,3].reshape((y.shape[0],y.shape[1],1))) )
      mse4.name = 'mse4'
      mae4.name = 'mae4'

      theta_mu4_in = theta_mu4_temp.reshape((x_shape[0]*x_shape[1], -1))
      theta_sig4_in = theta_sig4_temp.reshape((x_shape[0]*x_shape[1], -1))
      coeff4_in = coeff4_temp.reshape((x_shape[0]*x_shape[1], -1))

      argsGMM = argsGMM + (theta_mu4_in, theta_sig4_in, coeff4_in)
      ddoutMSEA = ddoutMSEA + [mse4, mae4]
      ddoutYpreds = ddoutYpreds + [y_pred4_temp]
      #totaMSE+=mse4
      indexSepDynamic +=2

    totaMSE = (mse1+mse2+mse3+mse4)/y_dim
    totaMSE.name = 'mse'

    totaMAE = (mae1+mae2+mae3+mae4)/y_dim
    totaMAE.name = 'mae'

    recon = GMMdisagMulti(y_dim, y_in, theta_mu1_in, theta_sig1_in, coeff1_in, *argsGMM)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon = recon.reshape((x_shape[0], x_shape[1]))
    recon.name = 'gmm_out'

    recon_term = recon.sum(axis=0).mean()
    recon_term = recon.sum(axis=0).mean()
    recon_term.name = 'recon_term'

    #kl_temp = kl_temp * mask
    
    kl_term = kl_temp.sum(axis=0).mean()
    kl_term.name = 'kl_term'

    #nll_upper_bound_0 = recon_term + kl_term
    #nll_upper_bound_0.name = 'nll_upper_bound_0'
    if (flgMSE==1):
      nll_upper_bound =  recon_term + kl_term + totaMSE
    else:
      nll_upper_bound =  recon_term + kl_term
    nll_upper_bound.name = 'nll_upper_bound'


    ######################

    model.inputs = [x, mask, y, y_mask, scheduleSamplingMask]
    model.params = params
    model.nodes = nodes

    optimizer = Adam(
        lr=lr
    )
    header = "epoch,log,kl,nll_upper_bound,mse,mae\n"
    extension = [
        GradientClipping(batch_size=batch_size),
        EpochCount(epoch, save_path, header),
        Monitoring(freq=monitoring_freq,
                   ddout=[nll_upper_bound, recon_term, kl_term,totaMSE, totaMAE, mse1, mae1]+ddoutMSEA+ddoutYpreds ,
                   indexSep=indexSepDynamic,
                   indexDDoutPlot = [13], # adding indexes of ddout for the plotting
                   #, (6,y_pred_temp)
                   instancesPlot = instancesPlot,#0-150
                   data=[Iterator(valid_data, batch_size)],
                   savedFolder = save_path),
        Picklize(freq=monitoring_freq, path=save_path),
        EarlyStopping(freq=monitoring_freq, path=save_path, channel=channel_name),
        WeightNorm()
    ]

    mainloop = Training(
        name=pkl_name,
        data=Iterator(train_data, batch_size),
        model=model,
        optimizer=optimizer,
        cost=nll_upper_bound,
        outputs=[recon_term, kl_term,nll_upper_bound,totaMSE,totaMAE],
        n_steps = n_steps,
        extension=extension,
        lr_iterations=lr_iterations,
        k_speedOfconvergence = kSchedSamp
    )
    
    mainloop.run()

    '''
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
    numBatchTest = 0
    for batch in data:
      outputGeneration = test_fn(batch[0], batch[2])
      testOutput.append(outputGeneration[1:12]) #before 36 including unnormalized metrics
      testMetrics2.append(outputGeneration[12:])
      #{0:[4,20], 2:[5,10]} 
      #if (numBatchTest==0):

      plt.figure(1)
      plt.plot(np.transpose(outputGeneration[0],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_Pred_0-4".format(numBatchTest))
      plt.clf()

      plt.figure(2)
      plt.plot(np.transpose(batch[2],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_RealDisag_0-4".format(numBatchTest))
      plt.clf()

      plt.figure(3)
      plt.plot(np.transpose(batch[0],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_Realagg_0-4".format(numBatchTest))
      plt.clf()
      numBatchTest+=1

    testOutput = np.asarray(testOutput)
    testMetrics2 = np.asarray(testMetrics2)
    print(testOutput.shape)
    print(testMetrics2.shape)

    testOutput[:,19:] = 1000 * testOutput[:,19:] # kwtts a watts
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


    print(testOutput[:,3:11].mean(),testOutput[:,11:19].mean())

    relErr1_test = testMetrics2[:,0].mean()
    relErr2_test = testMetrics2[:,1].mean()
    relErr3_test = testMetrics2[:,2].mean()
    relErr4_test = testMetrics2[:,3].mean()

    propAssigned1_test = testMetrics2[:, 8].mean()
    propAssigned2_test = testMetrics2[:, 9].mean()
    propAssigned3_test = testMetrics2[:, 10].mean()
    propAssigned4_test = testMetrics2[:, 11].mean()
    '''

    fLog = open(save_path+'/output.csv', 'w')
    fLog.write(str(lr_iterations)+"\n")
    fLog.write(str(appliances)+"\n")
    fLog.write(str(windows)+"\n\n")

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