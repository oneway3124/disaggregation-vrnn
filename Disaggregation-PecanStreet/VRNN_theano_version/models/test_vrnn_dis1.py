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

from cle.cle.cost import BiGMM, KLGaussianGaussian, GMM, GMM_outside
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

from VRNN_theano_version.datasets.dataport import Dataport
from VRNN_theano_version.datasets.dataport_utils import fetch_dataport

building = 2859
appliances = ['air1', 'furnace1', 'refrigerator1',  'clotheswasher1','drye1','dishwasher1', 'kitchenapp1', 'microwave1']
#windows = {building:("2015-01-01", "2015-01-01")}#, 2859:("2015-01-01", "2016-01-01"), 7951:("2015-01-01", "2016-01-01"),8292:("2015-01-01",  "2016-01-01"),3413:("2015-01-01", "2016-01-01")}#3413:("2015-01-01", "2015-12-31")
# dishwasher: windows = {6990:("2015-02-12", "2016-01-01"), 2859:("2015-02-01", "2015-12-15"), 7951:("2015-01-01", "2016-01-01"),8292:("2015-01-04",  "2016-01-01"),3413:("2015-01-01", "2015-12-25")}#3413:("2015-01-01", "2015-12-31")
# drye: windows = {2859:("2015-01-01", "2016-01-01"),6990:("2015-01-01", "2016-01-01"),7951:("2015-01-01", "2016-01-01"),8292:("2015-01-01",  "2016-01-01"),3413:("2015-01-20", "2015-12-31")}#3413:("2015-01-01", "2015-12-31")
windows = {2859:("2015-01-01", "2016-01-01"),6990:("2015-01-01", "2016-01-01"),7951:("2015-01-01", "2016-01-01"),8292:("2015-01-01",  "2016-01-01"),3413:("2015-01-01", "2016-01-01")}
listDates = {2859:['2015-08-26 07:57'],6990:['2015-10-15 08:18']}

def main(args):
    
    #theano.optimizer='fast_compile'
    #theano.config.exception_verbosity='high'

    trial = int(args['trial'])
    #pkl_name = 'dp_dis1-nosch_%d' % trial
    channel_name = 'mae'

    data_path = args['data_path']
    save_path = args['save_path'] #+'/gmm/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    testRef_file = args['testRef_file']
    flgMSE = int(args['flgMSE'])

    period = int(args['period'])
    n_steps = int(args['n_steps'])
    stride_train = int(args['stride_train'])
    stride_test = n_steps# int(args['stride_test'])

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
    typeLoad = int(args['typeLoad'])
    debug = int(args['debug'])
    kSchedSamp = int(args['kSchedSamp'])
    clipped = int(args['clipped'])

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "Reading pkl file '%s'" % testRef_file
    print "to the save path '%s'" % save_path

    q_z_dim = 150
    p_z_dim = 150
    p_x_dim = 150#250
    x2s_dim = 100#250
    y2s_dim = 100
    z2s_dim = 100#150
    target_dim = k#x_dim #(x_dim-1)*k

    model = Model()
    Xtrain, ytrain, Xval, yval, Xtest, ytest, reader = fetch_dataport(data_path, windows, appliances,numApps=flgAgg, period=period,
                                              n_steps= n_steps, stride_train = stride_train, stride_test = stride_test,
                                              trainPer=0.6, valPer=0.2, testPer=0.2, typeLoad=typeLoad,
                                              flgAggSumScaled = 1, flgFilterZeros = 1)
    print(reader.stdTrain, reader.meanTrain)
    instancesPlot = {0:[4], 2:[5]} #for now use hard coded instancesPlot for kelly sampling

    train_data = Dataport(name='train',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         inputX=Xtrain,
                         labels=ytrain)

    X_mean = train_data.X_mean
    X_std = train_data.X_std

    valid_data = Dataport(name='valid',
                         prep='normalize',
                         cond=True,# False
                         #path=data_path,
                         X_mean=X_mean,
                         X_std=X_std,
                         inputX=Xval,
                         labels = yval)

    test_data = Dataport(name='valid',
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

    x, mask, y , y_mask = test_data.theano_vars()
    scheduleSamplingMask  = T.fvector('schedMask')
    
    x.name = 'x_original'
    if debug:
        x.tag.test_value = np.zeros((15, batch_size, x_dim), dtype=np.float32)
        temp = np.ones((15, batch_size), dtype=np.float32)
        temp[:, -2:] = 0.
        mask.tag.test_value = temp

    fmodel = open(testRef_file, 'rb')
    mainloop = cPickle.load(fmodel)
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
    theta_mu = mainloop.model.nodes[11]
    theta_sig = mainloop.model.nodes[12]
    coeff = mainloop.model.nodes[13]


    nodes = [rnn,
             x_1, y_1, z_1, #dissag_pred,
             phi_1, phi_mu, phi_sig,
             prior_1, prior_mu, prior_sig,
             theta_1, theta_mu, theta_sig, coeff]#, corr, binary

    params = mainloop.model.params

    """params = OrderedDict()

    for node in nodes:
        if node.initialize() is not None:
            params.update(node.initialize())

    params = init_tparams(params)"""

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)

    def inner_fn_val(x_t, s_tm1):

        prior_1_t = prior_1.fprop([x_t,s_tm1], params)
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

        return s_t, prior_mu_t, prior_sig_t, theta_mu_t, theta_sig_t, coeff_t, pred_t#, y_pred
        #corr_temp, binary_temp
    ((s_temp_val, prior_mu_temp_val, prior_sig_temp_val,  theta_mu_temp_val, theta_sig_temp_val, coeff_temp_val, prediction_val), updates_val) =\
        theano.scan(fn=inner_fn_val,
                    sequences=[x_1_temp],
                    outputs_info=[s_0, None, None, None, None, None, None])

    for k, v in updates_val.iteritems():
        k.default_update = v

    s_temp_val = concatenate([s_0[None, :, :], s_temp_val[:-1]], axis=0)


    x_shape = x.shape

    ######################## TEST (GENERATION) TIME
    prediction_val = T.clip(prediction_val,0.0,np.inf)
    prediction_val.name = 'generated__'+str(flgAgg)
    mse_val = T.mean((prediction_val - y)**2) # As axis = None is calculated for all
    mae_val = T.mean( T.abs_(prediction_val - y) )

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

    data=Iterator(test_data, batch_size)

    test_fn = theano.function(inputs=[x,y],#[x, y],
                              allow_input_downcast=True,
                              outputs=[prediction_val, theta_mu_in_val, theta_sig_in_val, coeff_in_val, recon_term_val]#prediction_val, mse_val, mae_val
                              ,updates=updates_val#, allow_input_downcast=True, on_unused_input='ignore'
                              )
    testOutput = []
    bestInstsancesPred = []
    bestInstsancesDisa = []
    bestInstsancesAggr = []
    numBatchTest = 0
    for batch in data:
      #, recon_term_val, mse_val, mae_val
      x_test, y_test = batch[0], batch[2]
      outputGeneration = test_fn(x_test,y_test)#(20, 220, 1)
      #testOutput.append(outputGeneration[1:])
      # outputGeneration[0].shape #(20, 220, 40)

      if (clipped == 1):
        prediction_test = outputGeneration[0].clip(min=0)
      
      theta_mu_test = outputGeneration[1]
      theta_sig_test = outputGeneration[2]
      coeff_in_test = outputGeneration[3]

      realAggTest = np.transpose(x_test,[1,0,2])
      realDisagTest = np.transpose(y_test,[1,0,2]) # because y_test already transformed
      prediction_test =np.transpose(prediction_test,[1,0,2])

      ### test loglike
      y_test = y_test.reshape((y_test.shape[0]*y_test.shape[1],-1))
      recon_test = GMM_outside(y_test, theta_mu_test, theta_sig_test, coeff_in_test)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
      recon_test = recon_test.reshape((x_test.shape[0], x_test.shape[1]))
      recon_term_test= recon_test.sum(axis=0).mean()
  
      ## test mse - mae
      mse_test = np.mean((prediction_test - realDisagTest)**2) # As axis = None is calculated for all
      mae_test = np.mean(np.absolute(prediction_test - realDisagTest))
      testOutput.append((recon_term_test, mse_test, mae_test, outputGeneration[4]))

      batchMae_test = np.mean( np.absolute(prediction_test - realDisagTest), axis=(1,2) )
      idxMin = np.argmin(batchMae_test)

      for idx in np.asarray(idxMin).reshape(1,-1)[0,:]:

        plt.figure(1)
        plt.plot(prediction_test[idx])
        plt.plot(realDisagTest[idx])
        plt.legend(["Predicted","Real"])
        plt.savefig(save_path+"/vrnn_dis1_test-b{}_Pred-Real_0-{}".format(numBatchTest,idx))
        plt.clf()

        plt.figure(3)
        plt.plot(np.transpose(batch[0],[1,0,2])[idx])
        plt.savefig(save_path+"/vrnn_dis1_test-b{}_RealAgg_0-{}".format(numBatchTest,idx))
        plt.clf()

        bestInstsancesPred.append(prediction_test[idx])
        bestInstsancesDisa.append(realDisagTest[idx])
        bestInstsancesAggr.append(realAggTest[idx])

      numBatchTest+=1

    testOutput = np.asarray(testOutput)
    recon_testOutside = np.mean(testOutput[:, 0])
    mse_test =   np.mean(testOutput[:, 1])
    mae_test =   np.mean(testOutput[:, 2])
    recon_testInside = np.mean(testOutput[:, 3])
    #mseUnNorm_test =  testOutput[:, 3].mean()
    #maeUnNorm_test =  testOutput[:, 4].mean()

    fLog = open(save_path+'/output_test.csv', 'w')
    fLog.write(str(windows)+"\n")
    fLog.write("logTestOutside,logTestInside,mseTest,maeTest, mseTestUnNorm, maeTestUnNorm\n")
    fLog.write("{},{},{},{}\n".format(recon_testOutside, recon_testInside,mse_test,mae_test))
    fLog.write("q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim\n")
    fLog.write("{},{},{},{},{},{}\n".format(q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim))


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

    params['testRef_file'] = params['save_path']+'/gmmAE/'+params['testRef_file']
    params['save_path'] = params['save_path']+'/gmmAE/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")+'_app'+params['flgAgg']+'_test'
    os.makedirs(params['save_path'])
    shutil.copy('config_AE.txt', params['save_path']+'/config_AE.txt')

    main(params)
