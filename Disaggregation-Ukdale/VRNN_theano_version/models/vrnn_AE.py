import ipdb
import numpy as np
import theano
import theano.tensor as T
import datetime
import shutil
import os

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
windows = {1:("2013-02-27", "2017-02-27")}#, 2:("2013-02-27", "2013-04-27")
listDates = ['2013-08-26 07:57','2015-11-15 09:28']



def main(args):
    
    #theano.optimizer='fast_compile'
    #theano.config.exception_verbosity='high'

    trial = int(args['trial'])
    pkl_name = 'vrnn_gmm_%d' % trial
    channel_name = 'nll_upper_bound'

    data_path = args['data_path']
    save_path = args['save_path'] #+'/gmm/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    flgMSE = int(args['flgMSE'])

    genCase = int(args['genCase'])
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
    debug = int(args['debug'])
    num_sequences_per_batch = int(args['numSequences']) #based on appliance
    loadParam = args['loadAsKelly']
    target_inclusion_prob = float(args['target_inclusion_prob'])
    loadAsKelly = True
    if(loadParam == 'N' or loadParam == 'n' or loadParam == 'no' or loadParam == 'NO' or loadParam == 'No'):
        loadAsKelly = False

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path

    q_z_dim = 80#150
    p_z_dim = 80#150
    p_x_dim = 50#250
    x2s_dim = 50#250
    y2s_dim = 50
    z2s_dim = 50#150
    target_dim = k#x_dim #(x_dim-1)*k

    model = Model()
    Xtrain, ytrain, Xval, yval, reader = fetch_ukdale(data_path, windows, appliances,numApps=flgAgg, period=period,
                                              n_steps= n_steps, stride_train = stride_train, stride_test = stride_test,
                                              flgAggSumScaled = 1, flgFilterZeros = 1,
                                              isKelly= loadAsKelly, seq_per_batch=num_sequences_per_batch, target_inclusion_prob=target_inclusion_prob)
    
    instancesPlot = {0:[4,20], 2:[5,10]} #for now use hard coded instancesPlot for kelly sampling
    if(not loadAsKelly):
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


    init_W = InitCell('rand')
    init_U = InitCell('ortho')
    init_b = InitCell('zeros')
    init_b_sig = InitCell('const', mean=0.6)

    x, mask, y , y_mask = train_data.theano_vars()

    if (genCase ==1):
      inputX = x[:-1,:]
      targetX = x[1:,:]
      n_steps = n_steps-1
    else:
      inputX = x
      targetX = x  

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
               parent=['x_1', 'z_1'],
               parent_dim=[x2s_dim, z2s_dim],
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
                                  parent=['s_tm1'],
                                  parent_dim=[rnn_dim],
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
             x_1, y_1, z_1, #dissag_pred,
             phi_1, phi_mu, phi_sig,
             prior_1, prior_mu, prior_sig,
             theta_1, theta_mu, theta_sig, coeff]#, corr, binary

    params = OrderedDict()

    for node in nodes:
        if node.initialize() is not None:
            params.update(node.initialize())

    params = init_tparams(params)

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)
    y_1_temp = y_1.fprop([y], params)

    def inner_fn_train(x_t, y_t, s_tm1):

        phi_1_t = phi_1.fprop([x_t, s_tm1,y_t], params)
        phi_mu_t = phi_mu.fprop([phi_1_t], params)
        phi_sig_t = phi_sig.fprop([phi_1_t], params)

        prior_1_t = prior_1.fprop([s_tm1], params)
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
        s_t = rnn.fprop([[x_t, z_1_t], [s_tm1]], params)
        #y_pred = dissag_pred.fprop([s_t], params)

        return s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_t,  z_1_t, theta_1_t, theta_mu_t, theta_sig_t, coeff_t, pred#, y_pred
        #corr_temp, binary_temp
    ((s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp,z_t_temp, z_1_temp, theta_1_temp, theta_mu_temp, theta_sig_temp, coeff_temp, prediction), updates) =\
        theano.scan(fn=inner_fn_train,
                    sequences=[x_1_temp, y_1_temp],
                    outputs_info=[s_0, None, None, None, None, None, None,  None, None, None, None, None])

    
    for k, v in updates.iteritems():
        k.default_update = v
    
    s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)# seems like this is for creating an additional dimension to s_0

    s_temp.name = 'h_1'#gisse
    z_1_temp.name = 'z_1'#gisse
    z_t_temp.name = 'z'
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

    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

    x_shape = x.shape
    
    theta_mu_in = theta_mu_temp.reshape((x_shape[0]*x_shape[1], -1))
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
    nll_upper_bound.name = 'nll_upper_bound'

    model.inputs = [x, mask, y, y_mask]
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
                   ddout=[nll_upper_bound, recon_term, kl_term, mse, mae,
                          theta_mu_temp, z_t_temp, prediction,s_temp],
                   indexSep=5,
                   instancesPlot = instancesPlot, #{0:[4,20],2:[5,10]},#, 80,150
                   data=[Iterator(valid_data, batch_size)],
                   savedFolder = save_path),
        Picklize(freq=monitoring_freq, path=save_path),
        EarlyStopping(freq=monitoring_freq, path=save_path, channel=channel_name),
        WeightNorm()
    ]

    lr_iterations = {0:lr, 40:(lr/10),200:(lr/100)}

    mainloop = Training(
        name=pkl_name,
        data=Iterator(train_data, batch_size),
        model=model,
        optimizer=optimizer,
        cost=nll_upper_bound,
        outputs=[recon_term, kl_term, nll_upper_bound, mse, mae],
        extension=extension,
        lr_iterations=lr_iterations

    )
    mainloop.run()
    fLog = open(save_path+'/output.csv', 'w')
    fLog.write(str(lr_iterations)+"\n")
    fLog.write(str(windows)+"\n")
    fLog.write("q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim\n")
    fLog.write("{},{},{},{},{},{}\n".format(q_z_dim,p_z_dim,p_x_dim,x2s_dim,y2s_dim,z2s_dim))
    header = "epoch,log,mse,mae\n"
    fLog.write(header)
    for i , item in enumerate(mainloop.trainlog.monitor['recon_term']):
      f = mainloop.trainlog.monitor['epoch'][i]
      a = mainloop.trainlog.monitor['recon_term'][i]
      d = mainloop.trainlog.monitor['mse'][i]
      e = mainloop.trainlog.monitor['mae'][i]
      fLog.write("{},{},{},{}\n".format(f,a,d,e))
    


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
