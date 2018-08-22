import ipdb
import numpy as np
import theano
import theano.typed_list as TL
import theano.tensor as T
import datetime
import shutil
import os

from cle.cle.cost import BiGMM, KLGaussianGaussian, GMM,GMMdisag2
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

from preprocessing.ukdale import UKdale
from theano.sandbox.rng_mrg import MRG_RandomStreams
from ukdale_utils import plot_lines_iamondb_example

seed_rng = np.random.RandomState(np.random.randint(1024))
theano_seed = seed_rng.randint(np.iinfo(np.int32).max)
default_theano_rng = MRG_RandomStreams(theano_seed)

def main(args):
    
    theano.optimizer='fast_compile'
    #theano.config.exception_verbosity='high'
    

    trial = int(args['trial'])
    pkl_name = 'vrnn_gmm_%d' % trial
    channel_name = 'valid_nll_upper_bound'

    data_path = args['data_path']
    save_path = args['save_path']#+'/aggVSdisag_distrib/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    period = int(args['period'])
    n_steps = int(args['n_steps'])
    stride_train = int(args['stride_train'])
    stride_test = int(args['stride_test'])

    monitoring_freq = int(args['monitoring_freq'])
    epoch = int(args['epoch'])
    batch_size = int(args['batch_size'])
    x_dim = int(args['x_dim'])
    y_dim = int(args['y_dim'])
    z_dim = int(args['z_dim'])
    rnn_dim = int(args['rnn_dim'])
    k = int(args['num_k']) #a mixture of K Gaussian functions
    lr = float(args['lr'])
    debug = int(args['debug'])

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path

    q_z_dim = 12#150
    p_z_dim = 12#150
    p_x_dim = 20#250
    x2s_dim = 15#250
    z2s_dim = 20#150
    target_dim = k# As different appliances are separeted in theta_mu1, theta_mu2, etc... each one is just created from k different Gaussians

    model = Model()
    train_data = UKdale(name='train',
                         prep='normalize',
                         cond=True,# False
                         path=data_path,
                         numApps = y_dim,
                         period= period,
                         n_steps = n_steps,
                         x_dim=x_dim,
                         stride_train = stride_train,
                         stride_test = stride_test)

    X_mean = train_data.X_mean
    X_std = train_data.X_std

    valid_data = UKdale(name='valid',
                         prep='normalize',
                         cond=True,# False
                         path=data_path,
                         X_mean=X_mean,
                         X_std=X_std,
                         numApps = y_dim,
                         period= period,
                         n_steps = n_steps,
                         x_dim=x_dim,
                         stride_train = stride_train,
                         stride_test = stride_test)


    init_W = InitCell('rand')
    init_U = InitCell('ortho')
    init_b = InitCell('zeros')
    init_b_sig = InitCell('const', mean=0.6)

    x, mask, y , y_mask = train_data.theano_vars()
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
                                parent=['x_1', 's_tm1'],
                                parent_dim=[x2s_dim, rnn_dim],
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

    theta_mu1 = FullyConnectedLayer(name='theta_mu1',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_mu2 = FullyConnectedLayer(name='theta_mu2',
                                   parent=['theta_1'],
                                   parent_dim=[p_x_dim],
                                   nout=target_dim,
                                   unit='linear',
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

    theta_sig2 = FullyConnectedLayer(name='theta_sig2',
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

    coeff2 = FullyConnectedLayer(name='coeff2',
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
             x_1, z_1, #dissag_pred,
             phi_1, phi_mu, phi_sig,
             prior_1, prior_mu, prior_sig,
             theta_1, theta_mu1, theta_mu2, theta_sig1, theta_sig2, coeff1, coeff2]

    params = OrderedDict()

    for node in nodes:
        if node.initialize() is not None:
            params.update(node.initialize())

    params = init_tparams(params)

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)


    def inner_fn(x_t, s_tm1):

        phi_1_t = phi_1.fprop([x_t, s_tm1], params)
        phi_mu_t = phi_mu.fprop([phi_1_t], params)
        phi_sig_t = phi_sig.fprop([phi_1_t], params)

        prior_1_t = prior_1.fprop([s_tm1], params)
        prior_mu_t = prior_mu.fprop([prior_1_t], params)
        prior_sig_t = prior_sig.fprop([prior_1_t], params)

        z_t = Gaussian_sample(phi_mu_t, phi_sig_t)#in the original code it is gaussian. GMM is for the generation
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)

        
        theta_mu1_t = theta_mu1.fprop([theta_1_t], params)
        theta_sig1_t = theta_sig1.fprop([theta_1_t], params)
        coeff1_t = coeff1.fprop([theta_1_t], params)

        theta_mu2_t = theta_mu2.fprop([theta_1_t], params)
        theta_sig2_t = theta_sig2.fprop([theta_1_t], params)
        coeff2_t = coeff2.fprop([theta_1_t], params)
        
        #corr_t = corr.fprop([theta_1_t], params)
        #binary_t = binary.fprop([theta_1_t], params)

        y_pred1 = GMM_sampleY(theta_mu1_t, theta_sig1_t, coeff1_t) #Gaussian_sample(theta_mu_t, theta_sig_t)
        y_pred2 = GMM_sampleY(theta_mu2_t, theta_sig2_t, coeff2_t)        
        
        #y_pred = [GMM_sampleY(theta_mu_t[i], theta_sig_t[i], coeff_t[i]) for i in range(y_dim)]#T.stack([y_pred1,y_pred2],axis = 0 )
        s_t = rnn.fprop([[x_t, z_1_t], [s_tm1]], params)
        #y_pred = dissag_pred.fprop([s_t], params)

        return s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_t,  z_1_t, theta_1_t, theta_mu1_t, theta_sig1_t, coeff1_t, theta_mu2_t, theta_sig2_t, coeff2_t,y_pred1, y_pred2
    ((s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp,z_t_temp, z_1_temp, theta_1_temp, theta_mu1_temp, theta_sig1_temp, coeff1_temp, theta_mu2_temp, theta_sig2_temp, coeff2_temp, y_pred1_temp, y_pred2_temp), updates) =\
        theano.scan(fn=inner_fn,
                    sequences=[x_1_temp],
                    outputs_info=[s_0,  None, None, None, None, None, None, None, None,None, None, None, None, None, None,None])

    for k, v in updates.iteritems():
        k.default_update = v

    s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)# seems like this is for creating an additional dimension to s_0
    '''
    theta_1_temp = theta_1.fprop([z_1_temp, s_temp], params)
    theta_mu_temp = theta_mu.fprop([theta_1_temp], params)
    theta_sig_temp = theta_sig.fprop([theta_1_temp], params)
    coeff_temp = coeff.fprop([theta_1_temp], params)
    corr_temp = corr.fprop([theta_1_temp], params)
    binary_temp = binary.fprop([theta_1_temp], params)
    '''

    s_temp.name = 'h_1'#gisse
    z_1_temp.name = 'z_1'#gisse
    z_t_temp.name = 'z'

    #y_pred_temp.name = 'disaggregation'

    
    theta_mu1_temp.name = 'theta_mu1'
    theta_sig1_temp.name = 'theta_sig1'
    coeff1_temp.name = 'coeff1'

    theta_mu2_temp.name = 'theta_mu2'
    theta_sig2_temp.name = 'theta_sig2'
    coeff2_temp.name = 'coeff2'
    #corr_temp.name = 'corr'
    #binary_temp.name = 'binary'
    #x_pred_temp.name = 'x_reconstructed'
    y_pred1_temp.name = 'disaggregation1'
    y_pred2_temp.name = 'disaggregation2'
    y_pred_temp = T.stack([y_pred1_temp, y_pred2_temp], axis=2)
    y_pred_temp.name = 'disaggregation' 
    y_pred_temp = y_pred_temp.flatten(3)#dimshuffle(3,0,1,2)
    mse = T.mean((y_pred_temp - y.reshape((y.shape[0], y.shape[1],-1)))**2) # cause mse can be 26000
    #mse.name = 'mse'
    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

    x_shape = x.shape
    y_shape = y.shape
    x_in = x.reshape((x_shape[0]*x_shape[1], -1))
    y_in = y.reshape((y_shape[0]*y_shape[1], -1))
    
    theta_mu1_in = theta_mu1_temp.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig1_in = theta_sig1_temp.reshape((x_shape[0]*x_shape[1], -1))
    coeff1_in = coeff1_temp.reshape((x_shape[0]*x_shape[1], -1))

    theta_mu2_in = theta_mu2_temp.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig2_in = theta_sig2_temp.reshape((x_shape[0]*x_shape[1], -1))
    coeff2_in = coeff2_temp.reshape((x_shape[0]*x_shape[1], -1))

    #corr_in = corr_temp.reshape((x_shape[0]*x_shape[1], -1))
    #binary_in = binary_temp.reshape((x_shape[0]*x_shape[1], -1))

    recon = GMMdisag2(y_in, theta_mu1_in, theta_sig1_in, coeff1_in, theta_mu2_in, theta_sig2_in, coeff2_in)# BiGMM(x_in, theta_mu_in, theta_sig_in, coeff_in, corr_in, binary_in)
    recon = recon.reshape((x_shape[0], x_shape[1]))
    recon.name = 'gmm_out'
    
    #recon = recon * mask
    
    recon_term = recon.sum(axis=0).mean()
    recon_term.name = 'recon_term'

    #kl_temp = kl_temp * mask
    
    kl_term = kl_temp.sum(axis=0).mean()
    kl_term.name = 'kl_term'

    #nll_upper_bound_0 = recon_term + kl_term
    #nll_upper_bound_0.name = 'nll_upper_bound_0'
    nll_upper_bound =  recon_term + kl_term #+ mse
    nll_upper_bound.name = 'nll_upper_bound'

    '''
    max_x = x.max()
    mean_x = x.mean()
    min_x = x.min()
    max_x.name = 'max_x'
    mean_x.name = 'mean_x'
    min_x.name = 'min_x'

    max_theta_mu = theta_mu_in.max()
    mean_theta_mu = theta_mu_in.mean()
    min_theta_mu = theta_mu_in.min()
    max_theta_mu.name = 'max_theta_mu'
    mean_theta_mu.name = 'mean_theta_mu'
    min_theta_mu.name = 'min_theta_mu'

    max_theta_sig = theta_sig_in.max()
    mean_theta_sig = theta_sig_in.mean()
    min_theta_sig = theta_sig_in.min()
    max_theta_sig.name = 'max_theta_sig'
    mean_theta_sig.name = 'mean_theta_sig'
    min_theta_sig.name = 'min_theta_sig'

    coeff_max = coeff_in.max()
    coeff_min = coeff_in.min()
    coeff_mean_max = coeff_in.mean(axis=0).max()
    coeff_mean_min = coeff_in.mean(axis=0).min()
    coeff_max.name = 'coeff_max'
    coeff_min.name = 'coeff_min'
    coeff_mean_max.name = 'coeff_mean_max'
    coeff_mean_min.name = 'coeff_mean_min'

    max_phi_sig = phi_sig_temp.max()
    mean_phi_sig = phi_sig_temp.mean()
    min_phi_sig = phi_sig_temp.min()
    max_phi_sig.name = 'max_phi_sig'
    mean_phi_sig.name = 'mean_phi_sig'
    min_phi_sig.name = 'min_phi_sig'

    max_prior_sig = prior_sig_temp.max()
    mean_prior_sig = prior_sig_temp.mean()
    min_prior_sig = prior_sig_temp.min()
    max_prior_sig.name = 'max_prior_sig'
    mean_prior_sig.name = 'mean_prior_sig'
    min_prior_sig.name = 'min_prior_sig'
    '''
    model.inputs = [x, mask, y, y_mask]
    model.params = params
    model.nodes = nodes

    optimizer = Adam(
        lr=lr
    )

    extension = [
        GradientClipping(batch_size=batch_size),
        EpochCount(epoch),
        Monitoring(freq=monitoring_freq,
                   ddout=[nll_upper_bound, recon_term, kl_term, mse,#2
                          theta_1_temp, theta_mu1_temp, theta_sig1_temp, theta_mu2_temp, theta_sig2_temp,
                          z_t_temp, y_pred_temp,
                          coeff1_temp, coeff2_temp, s_temp, z_1_temp],
                   indexSep=4,
                   indexDDoutPlot = [(0,theta_mu1_temp), (5, z_t_temp), (6, y_pred_temp)], # adding indexes of ddout for the plotting
                   #, (6,y_pred_temp)
                   instancesPlot = [0, 150],#0-150
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
        outputs=[nll_upper_bound],
        extension=extension
    )
    mainloop.run()
    fLog = open(save_path+'/output.csv', 'w')
    print('Printing')
    print (len(mainloop.trainlog.monitor['nll_upper_bound']))
    fLog.write("log,kl,nll_upper_bound,mse\n")
    for i , item in enumerate(mainloop.trainlog.monitor['nll_upper_bound']):
      a = mainloop.trainlog.monitor['recon_term'][i]
      b = mainloop.trainlog.monitor['kl_term'][i]
      c = mainloop.trainlog.monitor['mse'][i]
      d = mainloop.trainlog.monitor['nll_upper_bound'][i]
      fLog.write("{},{},{},{}\n".format(a,b,d,c))
    fLog.close()

if __name__ == "__main__":

    import sys, time
    if len(sys.argv) > 1:
        config_file_name = sys.argv[-1]
    else:
        config_file_name = 'config.txt'

    f = open(config_file_name, 'r')
    lines = f.readlines()
    params = OrderedDict()

    for line in lines:
        line = line.split('\n')[0]
        param_list = line.split(' ')
        param_name = param_list[0]
        param_value = param_list[1]
        params[param_name] = param_value

    params['save_path'] = params['save_path']+'/aggVSdisag_distrib2/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    os.makedirs(params['save_path'])
    shutil.copy('config.txt', params['save_path']+'/config.txt')

    main(params)