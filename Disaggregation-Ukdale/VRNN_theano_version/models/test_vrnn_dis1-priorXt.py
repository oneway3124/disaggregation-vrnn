#import ipdb
import numpy as np
import theano
import theano.tensor as T
import datetime
import shutil
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
import cPickle

from cle.cle.cost import BiGMM, KLGaussianGaussian, GMM
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
from cle.cle.utils.op import Gaussian_sample, GMM_sample
from cle.cle.utils.gpu_op import concatenate

from preprocessing.ukdale import UKdale
from preprocessing.ukdale_utils import fetch_ukdale

appliances = [ 'kettle','microwave', 'washing machine', 'dish washer' , 'fridge']#
#windows = {1:("2013-02-27", "2017-02-27")}#, 2:("2013-02-27", "2013-04-27")
windows = {'train': {1: ("2013-02-27", "2015-02-27")}, 'test':{1:("2015-02-27", "2016-02-27")}, 'val':{1:("2016-02-27", "2017-02-27")}}
listDates = ['2013-08-26 07:57','2015-11-15 09:28']



def main(args):
    
    #theano.optimizer='fast_compile'
    #theano.config.exception_verbosity='high'

    trial = int(args['trial'])
    pkl_name = 'vrnn_gmm_%d' % trial
    channel_name = 'mse'

    data_path = args['data_path']
    save_path = args['save_path'] #+'/gmm/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    flgMSE = int(args['flgMSE'])

    genCase = int(args['genCase'])
    period = int(args['period'])
    n_steps = int(args['n_steps'])
    stride_train = int(args['stride_train'])
    stride_test = n_steps#int(args['stride_test'])

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


    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path

    q_z_dim = 150
    p_z_dim = 150
    p_x_dim = 200
    x2s_dim = 100
    y2s_dim = 100
    z2s_dim = 100
    target_dim = k#x_dim #(x_dim-1)*k

    model = Model()
    Xtrain, ytrain, Xval, yval, Xtest, ytest, reader = fetch_ukdale(data_path, windows, appliances,numApps=flgAgg, period=period,
                                              n_steps= n_steps, stride_train = stride_train, stride_test = stride_test,
                                              typeLoad= typeLoad, flgAggSumScaled = 1, flgFilterZeros = 1,
                                              seq_per_batch=num_sequences_per_batch, target_inclusion_prob=target_inclusion_prob)
    
    instancesPlot = {0:[10], 2:[5]} #for now use hard coded instancesPlot for kelly sampling
    if(typeLoad==0):
      instancesPlot = reader.build_dict_instances_plot(listDates, batch_size, Xval.shape[0])

    train_data = UKdale(name='train',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         inputX=Xtrain,
                         labels=ytrain)

    X_mean = train_data.X_mean
    X_std = train_data.X_std

    valid_data = UKdale(name='valid',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         X_mean=X_mean,
                         X_std=X_std,
                         inputX=Xval,
                         labels = yval)

    test_data = UKdale(name='valid',
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
    scheduleSamplingMask  = T.fvector('schedMask')
 
    x.name = 'x_original'
    if debug:
        x.tag.test_value = np.zeros((15, batch_size, x_dim), dtype=np.float32)
        temp = np.ones((15, batch_size), dtype=np.float32)
        temp[:, -2:] = 0.
        mask.tag.test_value = temp

    """x_1 = FullyConnectedLayer(name='x_1',
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
               parent=['x_1', 'z_1','y_1'],
               parent_dim=[x2s_dim, z2s_dim, y_dim],
               nout=rnn_dim,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

    phi_1 = FullyConnectedLayer(name='phi_1',
                                parent=['x_1', 's_tm1','y_1'],
                                parent_dim=[x2s_dim, rnn_dim,y2s_dim],
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
                                  parent=['x_1', 's_tm1'],
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

    theta_mu = FullyConnectedLayer(name='theta_mu',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_sig = FullyConnectedLayer(name='theta_sig',
                                    parent=['theta_1'],
                                    parent_dim=[p_x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    coeff = FullyConnectedLayer(name='coeff',
                                parent=['theta_1'],
                                parent_dim=[p_x_dim],
                                nout=k,
                                unit='softmax',
                                init_W=init_W,
                                init_b=init_b)"""

    fmodel = open('vrnn_gmm_dis1_best.pkl', 'rb')
    mainloop = cPickle.load(fmodel)
    fmodel.close()

    #attrs = vars(mainloop)
    #print ', '.join("%s: %s" % item for item in attrs.items())
    """names = [x.name for x in mainloop.model.nodes]
    print(names)
    print(mainloop.model.nodes)"""

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
    theta_mu = mainloop.model.nodes[11]
    theta_sig = mainloop.model.nodes[12]
    coeff = mainloop.model.nodes[13]

    nodes = [rnn,
         x_1, y_1, z_1, #dissag_pred,
         phi_1, phi_mu, phi_sig,
         prior_1, prior_mu, prior_sig,
         theta_1, theta_mu, theta_sig, coeff]#, corr, binary

    params = mainloop.model.params

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)
    y_1_temp = y_1.fprop([y], params)

    def inner_fn_val(x_t, s_tm1):

        prior_1_t = prior_1.fprop([x_t, s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(prior_mu_t, prior_sig_t)
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)
        theta_mu_t = theta_mu.fprop([theta_1_t], params)
        theta_sig_t = theta_sig.fprop([theta_1_t], params)

        coeff_t = coeff.fprop([theta_1_t], params)

        pred_t = GMM_sample(theta_mu_t, theta_sig_t, coeff_t) #Gaussian_sample(theta_mu_t, theta_sig_t)
        pred_1_t = y_1.fprop([pred_t], params)
        s_t = rnn.fprop([[x_t, z_1_t, pred_1_t], [s_tm1]], params)
        #y_pred = dissag_pred.fprop([s_t], params)

        return s_t, prior_mu_t, prior_sig_t, z_t,  z_1_t, theta_1_t, theta_mu_t, theta_sig_t, coeff_t, pred_t#, y_pred
        #corr_temp, binary_temp
    ((s_temp_val, prior_mu_temp_val, prior_sig_temp_val, z_t_temp_val, z_1_temp_val, theta_1_temp_val, theta_mu_temp_val, theta_sig_temp_val, coeff_temp_val, prediction_val), updates_val) =\
        theano.scan(fn=inner_fn_val,
                    sequences=[x_1_temp],
                    outputs_info=[s_0, None, None, None, None, None, None,  None, None, None])

    for k, v in updates_val.iteritems():
        k.default_update = v

    s_temp_val = concatenate([s_0[None, :, :], s_temp_val[:-1]], axis=0)

    """def inner_fn_train(x_t, y_t, s_tm1):

        phi_1_t = phi_1.fprop([x_t, s_tm1,y_t], params)
        phi_mu_t = phi_mu.fprop([phi_1_t], params)
        phi_sig_t = phi_sig.fprop([phi_1_t], params)

        prior_1_t = prior_1.fprop([x_t,s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(phi_mu_t, phi_sig_t)
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)
        theta_mu_t = theta_mu.fprop([theta_1_t], params)
        theta_sig_t = theta_sig.fprop([theta_1_t], params)

        coeff_t = coeff.fprop([theta_1_t], params)
        #corr_t = corr.fprop([theta_1_t], params)
        #binary_t = binary.fprop([theta_1_t], params)

        pred = GMM_sample(theta_mu_t, theta_sig_t, coeff_t) #Gaussian_sample(theta_mu_t, theta_sig_t)
        s_t = rnn.fprop([[x_t, z_1_t, y_t], [s_tm1]], params)
        #y_pred = dissag_pred.fprop([s_t], params)

        return s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, theta_mu_t, theta_sig_t, coeff_t, pred#, y_pred
        #corr_temp, binary_temp
    ((s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp, theta_mu_temp, theta_sig_temp, coeff_temp, prediction), updates) =\
        theano.scan(fn=inner_fn_train,
                    sequences=[x_1_temp, y_1_temp],
                    outputs_info=[s_0, None, None, None, None, None, None, None, None])

    
    for k, v in updates.iteritems():
        k.default_update = v
    
    #s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)# seems like this is for creating an additional dimension to s_0

    theta_mu_temp.name = 'theta_mu_temp'
    theta_sig_temp.name = 'theta_sig_temp'
    coeff_temp.name = 'coeff'

    if (flgAgg == -1 ):
      prediction.name = 'x_reconstructed'
      mse = T.mean((prediction - x)**2) # CHECK RESHAPE with an assertion
      mae = T.mean( T.abs(prediction - x) )
      mse.name = 'mse'
      pred_in = x.reshape((x_shape[0]*x_shape[1], -1))
    else:
      prediction.name = 'pred_'+str(flgAgg)
      mse = T.mean((prediction - y)**2) # As axis = None is calculated for all
      mae = T.mean( T.abs_(prediction - y) )
      mse.name = 'mse'
      mae.name = 'mae'
      pred_in = y.reshape((y.shape[0]*y.shape[1],-1))

    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)"""

    x_shape = x.shape
    
    """theta_mu_in = theta_mu_temp.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig_in = theta_sig_temp.reshape((x_shape[0]*x_shape[1], -1))
    coeff_in = coeff_temp.reshape((x_shape[0]*x_shape[1], -1))
    #corr_in = corr_temp.reshape((x_shape[0]*x_shape[1], -1))
    #binary_in = binary_temp.reshape((x_shape[0]*x_shape[1], -1))

    recon = GMM(pred_in, theta_mu_in, theta_sig_in, coeff_in)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon = recon.reshape((x_shape[0], x_shape[1]))
    recon.name = 'gmm_out'

    recon_term = recon.sum(axis=0).mean()
    recon_term.name = 'recon_term'
    
    kl_term = kl_temp.sum(axis=0).mean()
    kl_term.name = 'kl_term'

    nll_upper_bound = recon_term + kl_term #+ mse
    if (flgMSE):
      nll_upper_bound = nll_upper_bound + mse
    nll_upper_bound.name = 'nll_upper_bound'"""

    ######################## TEST (GENERATION) TIME
    prediction_val.name = 'generated__'+str(flgAgg)
    mse_val = T.mean((prediction_val - y)**2) # As axis = None is calculated for all
    mae_val = T.mean( T.abs_(prediction_val - y))

    totPred = T.sum(prediction_val)
    totReal = T.sum(y)
    relErr_val =( totPred -  totReal)/ T.maximum(totPred,totReal)
    propAssigned_val = 1 - T.sum(T.abs_(prediction_val - y))/(2*T.sum(x))

    mse_val.name = 'mse_val'
    mae_val.name = 'mae_val'
    pred_in_val = y.reshape((y.shape[0]*y.shape[1],-1))

    theta_mu_in_val = theta_mu_temp_val.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig_in_val = theta_sig_temp_val.reshape((x_shape[0]*x_shape[1], -1))
    coeff_in_val = coeff_temp_val.reshape((x_shape[0]*x_shape[1], -1))

    recon_val = GMM(pred_in_val, theta_mu_in_val, theta_sig_in_val, coeff_in_val)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon_val = recon_val.reshape((x_shape[0], x_shape[1]))
    recon_val.name = 'gmm_out_val'

    recon_term_val= recon_val.sum(axis=0).mean()
    recon_term_val.name = 'recon_term_val'

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
                   #ddout=[nll_upper_bound, recon_term, kl_term, mse, mae,
                    #      theta_mu_temp,prediction],
                   indexSep=5,
                   instancesPlot = instancesPlot, #{0:[4,20],2:[5,10]},#, 80,150
                   data=[Iterator(valid_data, batch_size)],
                   savedFolder = save_path),
        Picklize(freq=monitoring_freq, path=save_path),
        EarlyStopping(freq=monitoring_freq, path=save_path, channel=channel_name),
        WeightNorm()
    ]

    lr_iterations = {0:lr, 75:(lr/10), 150:(lr/100)}

    """mainloop = Training(
        name=pkl_name,
        data=Iterator(train_data, batch_size),
        model=model,
        optimizer=optimizer,
        cost=nll_upper_bound,
        outputs=[recon_term, kl_term, nll_upper_bound, mse, mae],
        n_steps = n_steps,
        extension=extension,
        lr_iterations=lr_iterations

    )
    mainloop.run()"""

    data=Iterator(test_data, batch_size)

    test_fn = theano.function(inputs=[x, y],#[x, y],
                              #givens={x:Xtest},
                              #on_unused_input='ignore',
                              #z=( ,200,1)
                              allow_input_downcast=True,
                              outputs=[prediction_val, recon_term_val, mse_val, mae_val, relErr_val,propAssigned_val ]#prediction_val, mse_val, mae_val
                              ,updates=updates_val#, allow_input_downcast=True, on_unused_input='ignore'
                              )
    testOutput = []
    testMetrics2 = []
    numBatchTest = 0
    for batch in data:
      outputGeneration = test_fn(batch[0], batch[2])
      testOutput.append(outputGeneration[1:4])
      testMetrics2.append(outputGeneration[4:])
      #{0:[4,20], 2:[5,10]} 
      #if (numBatchTest==0):
      '''
      plt.figure(1)
      plt.plot(np.transpose(outputGeneration[0],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_z_0-4".format(numBatchTest))
      plt.clf()

      plt.figure(2)
      plt.plot(np.transpose(outputGeneration[1],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_s_0-4".format(numBatchTest))
      plt.clf()

      plt.figure(3)
      plt.plot(np.transpose(outputGeneration[2],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_theta_0-4".format(numBatchTest))
      plt.clf()
      '''
      plt.figure(4)
      plt.plot(np.transpose(outputGeneration[0],[1,0,2])[4])
      plt.plot(np.transpose(batch[2],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_RealAndPred_0-4".format(numBatchTest))
      plt.clf()

      plt.figure(4)
      plt.plot(np.transpose(batch[0],[1,0,2])[4])
      plt.savefig(save_path+"/vrnn_dis_generated{}_Realagg_0-4".format(numBatchTest))
      plt.clf()
      numBatchTest+=1

    testOutput = np.asarray(testOutput)
    testMetrics2 = np.asarray(testMetrics2)
    print(testOutput.shape)
    print(testMetrics2.shape)
    recon_test = testOutput[:, 0].mean()
    mse_test =  testOutput[:, 1].mean()
    mae_test =  testOutput[:, 2].mean()

    relErr_test = testMetrics2[:,0].mean()
    propAssig_test = testMetrics2[:,1].mean()

    fLog = open(save_path+'/output.csv', 'w')
    fLog.write(str(lr_iterations)+"\n")
    fLog.write(str(windows)+"\n")
    fLog.write("logTest,mseTest,maeTest,relError,propAssig\n")
    fLog.write("{},{},{}\n".format(recon_test,mse_test,mae_test,relErr_test, propAssig_test))
    fLog.write("q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim\n")
    fLog.write("{},{},{},{},{},{}\n".format(q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim))
    header = "epoch,log,kl,mse,mae\n"
    fLog.write(header)
    for i , item in enumerate(mainloop.trainlog.monitor['recon_term']):
      f = mainloop.trainlog.monitor['epoch'][i]
      a = mainloop.trainlog.monitor['recon_term'][i]
      b = mainloop.trainlog.monitor['kl_term'][i]
      d = mainloop.trainlog.monitor['mse'][i]
      e = mainloop.trainlog.monitor['mae'][i]
      fLog.write("{:d},{:.2f},{:.2f},{:.3f},{:.3f}\n".format(f,a,b,d,e))

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

    params['save_path'] = params['save_path']+'/gmmAE/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")+'_app'+params['flgAgg']
    os.makedirs(params['save_path'])
    shutil.copy('config_AE.txt', params['save_path']+'/config_AE.txt')

    main(params)