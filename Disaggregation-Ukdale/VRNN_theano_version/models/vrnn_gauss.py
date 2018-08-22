import ipdb
import numpy as np
import theano
import theano.tensor as T

from cle.cle.cost import BiGauss, KLGaussianGaussian, Gaussian
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
from cle.cle.utils.op import Gaussian_sample
from cle.cle.utils.gpu_op import concatenate

from preprocessing.ukdale import UKdale
from ukdale_utils import fetch_ukdale

appliances = [ 'kettle','microwave', 'washing machine', 'dish washer' , 'fridge']#
windows = {1:("2013-02-27", "2015-02-27")}#, 2:("2013-02-27", "2013-04-27")

def main(args):

    theano.optimizer='fast_compile'
    theano.config.exception_verbosity='high'
    trial = int(args['trial'])
    pkl_name = 'vrnn_gauss_%d' % trial
    channel_name = 'valid_nll_upper_bound'

    data_path = args['data_path']
    save_path = args['save_path']
    save_path = args['save_path']
    period = int(args['period'])
    n_steps = int(args['n_steps'])
    stride_train = int(args['stride_train'])
    stride_test = int(args['stride_test'])

    monitoring_freq = int(args['monitoring_freq'])
    epoch = int(args['epoch'])
    batch_size = int(args['batch_size'])
    x_dim = int(args['x_dim'])
    z_dim = int(args['z_dim'])
    rnn_dim = int(args['rnn_dim'])
    lr = float(args['lr'])
    debug = int(args['debug'])

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path


    q_z_dim = 150
    p_z_dim = 150
    p_x_dim = 250
    x2s_dim = 10#250
    z2s_dim = 10#150
    target_dim = x_dim#(x_dim-1)

    model = Model()
    Xtrain, ytrain, Xval, yval = fetch_ukdale(data_path, windows, appliances,numApps=flgAgg, period=period,n_steps= n_steps, stride_train = stride_train, stride_test = stride_test)
    
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

    x, y = train_data.theano_vars()

    if debug:
        x.tag.test_value = np.zeros((15, batch_size, x_dim), dtype=np.float32)
        temp = np.ones((15, batch_size), dtype=np.float32)
        temp[:, -2:] = 0.
        mask.tag.test_value = temp

    x_1 = FullyConnectedLayer(name='x_1',
                              parent=['x_t'], #OrderDict parent['x_t'] = x_dim
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

    phi_1 = FullyConnectedLayer(name='phi_1', ## encoder
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

    theta_1 = FullyConnectedLayer(name='theta_1', ### decoder
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

    corr = FullyConnectedLayer(name='corr',  ## rho
                               parent=['theta_1'],
                               parent_dim=[p_x_dim],
                               nout=1,
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
             x_1, z_1,
             phi_1, phi_mu, phi_sig,
             prior_1, prior_mu, prior_sig,
             theta_1, theta_mu, theta_sig] #, corr, binary

    params = OrderedDict()

    for node in nodes:
        if node.initialize() is not None:
            params.update(node.initialize()) #Initialize values of the W matrices according to dim of parents

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

        z_t = Gaussian_sample(phi_mu_t, phi_sig_t)
        z_1_t = z_1.fprop([z_t], params)

        theta_1_t = theta_1.fprop([z_1_t, s_tm1], params)
        theta_mu_t = theta_mu.fprop([theta_1_t], params)
        theta_sig_t = theta_sig.fprop([theta_1_t], params)

        pred = Gaussian_sample(theta_mu_t, theta_sig_t)

        s_t = rnn.fprop([[x_t, z_1_t], [s_tm1]], params)

        return s_t, phi_mu_t, phi_sig_t, prior_mu_t, prior_sig_t, z_t,  z_1_t, theta_1_t, theta_mu_t, theta_sig_t, pred

    ((s_temp, phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp, z_temp, z_1_temp, theta_1_temp, theta_mu_temp, theta_sig_temp, pred_temp), updates) =\
        theano.scan(fn=inner_fn,
                    sequences=[x_1_temp], #non_sequences unchanging variables
                    #The tensor(s) to be looped over should be provided to scan using the sequence keyword argument
                    outputs_info=[s_0, None, None, None, None, None, None, None, None, None, None])#Initialization occurs in outputs_info
        #=None This indicates to scan that it does not need to pass the prior result to _fn
    '''
    The general order of function parameters to:
    sequences (if any), prior result(s) (if needed), non-sequences (if any)
    '''
    for k, v in updates.iteritems():
        print("Update")
        k.default_update = v

    s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)
    s_temp.name = 'h_1'#gisse
    z_temp.name = 'z'
    z_1_temp.name = 'z_1'#gisse
    #theta_1_temp = theta_1.fprop([z_1_temp, s_temp], params)
    #theta_mu_temp = theta_mu.fprop([theta_1_temp], params)
    theta_mu_temp.name = 'theta_mu'
    #theta_sig_temp = theta_sig.fprop([theta_1_temp], params)
    theta_sig_temp.name = 'theta_sig'
    x_pred_temp.name = 'x_reconstructed'
    #corr_temp = corr.fprop([theta_1_temp], params)
    #corr_temp.name = 'corr'
    #binary_temp = binary.fprop([theta_1_temp], params)
    #binary_temp.name = 'binary'

    if (flgAgg == -1 ):
      prediction.name = 'x_reconstructed'
      mse = T.mean((prediction - x)**2) # CHECK RESHAPE with an assertion
      mae = T.mean( T.abs(prediction - x) )
      mse.name = 'mse'
      pred_in = x.reshape((x_shape[0]*x_shape[1], -1))
    else:
      prediction.name = 'pred_'+str(flgAgg)
      mse = T.mean((prediction - y[:,:,flgAgg].reshape((y.shape[0],y.shape[1],1)))**2) # CHECK RESHAPE with an assertion
      mae = T.mean( T.abs_(prediction - y[:,:,flgAgg].reshape((y.shape[0],y.shape[1],1))) )
      mse.name = 'mse'
      mae.name = 'mae'
      pred_in = y[:,:,flgAgg].reshape((x.shape[0]*x.shape[1],-1), ndim=2)


    kl_temp = KLGaussianGaussian(phi_mu_temp, phi_sig_temp, prior_mu_temp, prior_sig_temp)

    #x_shape = x.shape
    #x_in = x.reshape((x_shape[0]*x_shape[1], -1))
    theta_mu_in = theta_mu_temp.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig_in = theta_sig_temp.reshape((x_shape[0]*x_shape[1], -1))
    #corr_in = corr_temp.reshape((x_shape[0]*x_shape[1], -1))
    #binary_in = binary_temp.reshape((x_shape[0]*x_shape[1], -1))

    recon = Gaussian(pred_in, theta_mu_in, theta_sig_in) # BiGauss(x_in, theta_mu_in, theta_sig_in, corr_in, binary_in) # second term for the loss function
    recon = recon.reshape((x_shape[0], x_shape[1]))
    #recon = recon * mask
    recon_term = recon.sum(axis=0).mean()
    recon_term.name = 'recon_term'

    #kl_temp = kl_temp * mask
    kl_term = kl_temp.sum(axis=0).mean()
    kl_term.name = 'kl_term'

    nll_upper_bound = recon_term + kl_term
    nll_upper_bound.name = 'nll_upper_bound'

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

    prior_sig_output = prior_sig_temp
    prior_sig_output.name = 'prior_sig_o'
    phi_sig_output = phi_sig_temp
    phi_sig_output.name = 'phi_sig_o'

    model.inputs = [x, mask]
    model.params = params
    model.nodes = nodes

    optimizer = Adam(
        lr=lr
    )

    extension = [
        GradientClipping(batch_size=batch_size),
        EpochCount(epoch),
        Monitoring(freq=monitoring_freq,
                   ddout=[nll_upper_bound, recon_term, kl_term, mse, mae,
                          max_phi_sig, mean_phi_sig, min_phi_sig,
                          max_prior_sig, mean_prior_sig, min_prior_sig,
                          max_theta_sig, mean_theta_sig, min_theta_sig,
                          max_x, mean_x, min_x,
                          max_theta_mu, mean_theta_mu, min_theta_mu, #0-17
                          #binary_temp, corr_temp, 
                          theta_mu_temp, theta_sig_temp, #17-20
                          s_temp, z_temp, z_1_temp, x_pred_temp
                          #phi_sig_output,phi_sig_output
                          ],## added in order to explore the distributions
                   indexSep=22,
                   indexDDoutPlot = [(0,theta_mu_temp), (2, z_t_temp), (3,prediction)],
                   instancesPlot = [0,150],#, 80,150
                   savedFolder = save_path,
                   data=[Iterator(valid_data, batch_size)]),
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
    fLog.write("log,kl,nll_upper_bound,mse,mae\n")
    for i , item in enumerate(mainloop.trainlog.monitor['nll_upper_bound']):
      a = mainloop.trainlog.monitor['recon_term'][i]
      b = mainloop.trainlog.monitor['kl_term'][i]
      c = mainloop.trainlog.monitor['nll_upper_bound'][i]
      d = mainloop.trainlog.monitor['mse'][i]
      e = mainloop.trainlog.monitor['mae'][i]
      fLog.write("{},{},{},{},{}\n".format(a,b,c,d,e))

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

    params['save_path'] = params['save_path']+'/gauss/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    os.makedirs(params['save_path'])
    shutil.copy('config.txt', params['save_path']+'/config.txt')

    main(params)
