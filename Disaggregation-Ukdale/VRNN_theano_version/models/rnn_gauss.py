import ipdb
import numpy as np
import theano
import theano.tensor as T
import datetime
import shutil
import os

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
from cle.cle.utils.gpu_op import concatenate

from preprocessing.ukdale import UKdale

appliances = [ 'kettle','microwave', 'washing machine', 'dish washer' , 'fridge']#
windows = {1:("2013-02-27", "2014-02-27")}#, 2:("2013-02-27", "2013-04-27")

def main(args):

    trial = int(args['trial'])
    pkl_name = 'rnn_gauss_%d' % trial
    channel_name = 'valid_nll'

    data_path = args['data_path']
    save_path = args['save_path']
    flgMSE = int(args['flgMSE'])

    monitoring_freq = int(args['monitoring_freq'])
    epoch = int(args['epoch'])
    batch_size = int(args['batch_size'])
    x_dim = int(args['x_dim'])
    z_dim = int(args['z_dim'])
    y_dim = int(args['y_dim'])
    flgAgg = int(args['flgAgg'])
    rnn_dim = int(args['rnn_dim'])
    lr = float(args['lr'])
    debug = int(args['debug'])

    print "trial no. %d" % trial
    print "batch size %d" % batch_size
    print "learning rate %f" % lr
    print "saving pkl file '%s'" % pkl_name
    print "to the save path '%s'" % save_path

    x2s_dim = 340
    s2x_dim = 340
    target_dim = k#x_dim - 1

    model = Model()
    train_data = UKdale(name='train',
                         prep='normalize',
                         cond=False,
                         path=data_path,
                         windows = windows,
                         appliances = appliances,
                         numApps = flgAgg,
                         period= period,
                         n_steps = n_steps,
                         stride_train = stride_train,
                         stride_test = stride_test)

    X_mean = train_data.X_mean
    X_std = train_data.X_std

    valid_data = UKdale(name='valid',
                         prep='normalize',
                         cond=False,
                         path=data_path,
                         X_mean=X_mean,
                         X_std=X_std,
                         windows = windows,
                         appliances = appliances,
                         numApps = flgAgg,
                         period= period,
                         n_steps = n_steps,
                         stride_train = stride_train,
                         stride_test = stride_test)

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
                              parent=['x_t'],
                              parent_dim=[x_dim],
                              nout=x2s_dim,
                              unit='relu',
                              init_W=init_W,
                              init_b=init_b)

    rnn = LSTM(name='rnn',
               parent=['x_1'],
               parent_dim=[x2s_dim],
               nout=rnn_dim,
               unit='tanh',
               init_W=init_W,
               init_U=init_U,
               init_b=init_b)

    theta_1 = FullyConnectedLayer(name='theta_1',
                                  parent=['s_tm1'],
                                  parent_dim=[rnn_dim],
                                  nout=s2x_dim,
                                  unit='relu',
                                  init_W=init_W,
                                  init_b=init_b)

    theta_mu = FullyConnectedLayer(name='theta_mu',
                                   parent=['theta_1'],
                                   parent_dim=[s2x_dim],
                                   nout=target_dim,
                                   unit='linear',
                                   init_W=init_W,
                                   init_b=init_b)

    theta_sig = FullyConnectedLayer(name='theta_sig',
                                    parent=['theta_1'],
                                    parent_dim=[s2x_dim],
                                    nout=target_dim,
                                    unit='softplus',
                                    cons=1e-4,
                                    init_W=init_W,
                                    init_b=init_b_sig)

    corr = FullyConnectedLayer(name='corr',
                                parent=['theta_1'],
                                parent_dim=[s2x_dim],
                                nout=1,
                                unit='tanh',
                                init_W=init_W,
                                init_b=init_b)

    binary = FullyConnectedLayer(name='binary',
                                parent=['theta_1'],
                                parent_dim=[s2x_dim],
                                nout=1,
                                unit='sigmoid',
                                init_W=init_W,
                                init_b=init_b)

    nodes = [rnn, x_1, theta_1, theta_mu, theta_sig] #, corr, binary

    params = OrderedDict()

    for node in nodes:
        if node.initialize() is not None:
            params.update(node.initialize())

    params = init_tparams(params)

    s_0 = rnn.get_init_state(batch_size)

    x_1_temp = x_1.fprop([x], params)


    def inner_fn(x_t, s_tm1):

        s_t = rnn.fprop([[x_t], [s_tm1]], params)
        theta_1_t = theta_1.fprop([s_t], params)
        theta_mu_t = theta_mu.fprop([theta_1_t], params)
        theta_sig_t = theta_sig.fprop([theta_1_t], params)
        coeff_t = coeff.fprop([theta_1_t], params)

        pred = Gaussian_sample(theta_mu_t, theta_sig_t)
        return s_t, theta_mu_t, theta_sig_t, coeff_t, pred

    ((s_temp, theta_mu_temp, theta_sig_temp, coeff_temp,  pred_temp), updates) = theano.scan(fn=inner_fn,
                                      sequences=[x_1_temp],
                                      outputs_info=[s_0, None, None, None, None])

    for k, v in updates.iteritems():
        k.default_update = v

    s_temp = concatenate([s_0[None, :, :], s_temp[:-1]], axis=0)
    '''
    theta_1_temp = theta_1.fprop([s_temp], params)
    theta_mu_temp = theta_mu.fprop([theta_1_temp], params)
    theta_sig_temp = theta_sig.fprop([theta_1_temp], params)
    corr_temp = corr.fprop([theta_1_temp], params)
    binary_temp = binary.fprop([theta_1_temp], params)
    '''
    x_shape = x.shape
    x_in = x.reshape((x_shape[0]*x_shape[1], -1))
    theta_mu_in = theta_mu_temp.reshape((x_shape[0]*x_shape[1], -1))
    theta_sig_in = theta_sig_temp.reshape((x_shape[0]*x_shape[1], -1))
    corr_in = corr_temp.reshape((x_shape[0]*x_shape[1], -1))
    binary_in = binary_temp.reshape((x_shape[0]*x_shape[1], -1))

    if (flgAgg == -1 ):
      prediction.name = 'x_reconstructed'
      mse = T.mean((prediction - x)**2) # CHECK RESHAPE with an assertion
      mae = T.mean( T.abs(prediction - x) )
      mse.name = 'mse'
      pred_in = x.reshape((x_shape[0]*x_shape[1], -1))
    else:
      pred_temp = pred_temp.reshape((pred_temp.shape[0], pred_temp.shape[1]))
      pred_temp.name = 'pred_'+str(flgAgg)
      #y[:,:,flgAgg].reshape((y.shape[0],y.shape[1],1))
      mse = T.mean((pred_temp - y.T)**2) # CHECK RESHAPE with an assertion
      mae = T.mean( T.abs_(pred_temp - y.T))
      mse.name = 'mse'
      mae.name = 'mae'
      pred_in = y.reshape((x.shape[0]*x.shape[1],-1), ndim=2)

    recon = Gaussian(pred_in, theta_mu_in, theta_sig_in)
    recon = recon.reshape((x_shape[0], x_shape[1]))
    #recon = recon * mask
    recon_term = recon.sum(axis=0).mean()
    recon_term.name = 'nll'

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

    model.inputs = [x, y]
    model.params = params
    model.nodes = nodes

    optimizer = Adam(
        lr=lr
    )

    extension = [
        GradientClipping(batch_size=batch_size),
        EpochCount(epoch),
        Monitoring(freq=monitoring_freq,
                   ddout=[recon_term,
                          max_theta_sig, mean_theta_sig, min_theta_sig,
                          max_x, mean_x, min_x,
                          max_theta_mu, mean_theta_mu, min_theta_mu],
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
        cost=recon_term,
        outputs=[recon_term],
        extension=extension
    )
    mainloop.run()
    fLog = open(save_path+'/output.csv', 'w')
    fLog.write("log,mse,mae\n")
    for i , item in enumerate(mainloop.trainlog.monitor['nll_upper_bound']):
      a = mainloop.trainlog.monitor['recon_term'][i]
      d = mainloop.trainlog.monitor['mse'][i]
      e = mainloop.trainlog.monitor['mae'][i]
      fLog.write("{},{},{}\n".format(a,d,e))


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

    params['save_path'] = params['save_path']+'/rnn_gmm/'+datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    os.makedirs(params['save_path'])
    shutil.copy('config.txt', params['save_path']+'/config.txt')

    main(params)

