import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.utils.op import logsumexp


def NllBin(y, y_hat):
    """
    Binary cross-entropy

    Parameters
    ----------
    .. todo::
    """
    nll = T.nnet.binary_crossentropy(y_hat, y).sum(axis=-1)
    return nll


def NllMul(y, y_hat):
    """
    Multi cross-entropy

    Parameters
    ----------
    .. todo::
    """
    ll = (y * T.log(y_hat)).sum(axis=-1)
    nll = -ll
    return nll


def NllMulInd(y, y_hat):
    """
    Multi cross-entropy
    Efficient implementation using the indices in y

    Credit assignment:
    This code is brought from: https://github.com/lisa-lab/pylearn2

    Parameters
    ----------
    .. todo::
    """
    log_prob = T.log(y_hat)
    flat_log_prob = log_prob.flatten()
    flat_y = y.flatten()
    flat_indices = flat_y + T.arange(y.shape[0]) * log_prob.shape[1]
    ll = flat_log_prob[T.cast(flat_indices, 'int64')]
    nll = -ll
    return nll


def MSE(y, y_hat, use_sum=1):
    """
    Mean squared error

    Parameters
    ----------
    .. todo::
    """
    if use_sum:
        mse = T.sum(T.sqr(y - y_hat), axis=-1)
    else:
        mse = T.mean(T.sqr(y - y_hat), axis=-1)
    return mse


def Laplace(y, mu, sig):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    nll = T.sum(abs(y - mu) / sig + T.log(sig) + T.log(2), axis=-1)
    return nll


def Gaussian(y, mu, sig):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                      T.log(2 * np.pi), axis=-1)
    return nll


def GMM(y, mu, sig, coeff):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y = y.dimshuffle(0, 1, 'x')
    y.name = 'y_shuffled'
    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]//coeff.shape[-1],
                     coeff.shape[-1]))
    mu.name = 'mu'
    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]//coeff.shape[-1],
                       coeff.shape[-1]))
    sig.name = 'sig'
    a = T.sqr(y - mu)
    a.name = 'a'
    inner = -0.5 * T.sum(a / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner.name = 'inner'
    nll = -logsumexp(T.log(coeff) + inner, axis=1)
    nll.name = 'logsum'
    return nll

def GMM_outside(y, mu, sig, coeff):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable no, array
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y = np.expand_dims(y, axis=2)
    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]//coeff.shape[-1],
                     coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]//coeff.shape[-1],
                       coeff.shape[-1]))
    a = np.square(y - mu)
    inner = -0.5 * np.sum(a / sig**2 + 2 * np.log(sig) + np.log(2 * np.pi), axis=1)
    #print(inner.shape)#(steps*batch,k)
    aux1 = np.log(coeff) + inner # why not log(coef)+log(inner)??
    #print(aux1.shape)
    #nll = -logsumexp(np.log(coeff) + inner, axis=1)
    
    x_max = np.amax(aux1, axis=1, keepdims=True)
    z = np.log(np.sum(np.exp(aux1 - x_max), axis=1, keepdims=True)) + x_max
    nll =  np.sum(z,axis=1)

    return -nll

def GMMdisag2(y, mu, sig, coeff, mu2, sig2, coeff2):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y1 = y[:,0].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,0,:]
    y2 = y[:,1].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,1,:]
    y1.name = 'y1_shuffled'
    y2.name = 'y2_shuffled'
    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))

    mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
    sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))

    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'

    inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
    nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)
    nll = nll1 + nll2
    return nll

def GMMdisag3(y, mu, sig, coeff, mu2, sig2, coeff2, mu3, sig3, coeff3):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y1 = y[:,0].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,0,:]
    y2 = y[:,1].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,1,:]
    y3 = y[:,2].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
    y1.name = 'y1_shuffled'
    y2.name = 'y2_shuffled'
    y3.name = 'y3_shuffled'
    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))

    mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
    sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))

    mu3 = mu3.reshape((mu3.shape[0],mu3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))
    sig3 = sig3.reshape((sig3.shape[0],sig3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))


    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'

    inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
    nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)

    inner3 = -0.5 * T.sum(T.sqr(y3 - mu3) / sig3**2 + 2 * T.log(sig3) + T.log(2 * np.pi), axis=1)
    nll3 = -logsumexp(T.log(coeff3) + inner3, axis=1)

    nll = nll1 + nll2 + nll3
    return nll

def GMMdisag4(y, mu, sig, coeff, mu2, sig2, coeff2, mu3, sig3, coeff3, mu4, sig4, coeff4):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y1 = y[:,0].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,0,:]
    y2 = y[:,1].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,1,:]
    y3 = y[:,2].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
    y4 = y[:,3].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')

    y1.name = 'y1_shuffled'
    y2.name = 'y2_shuffled'
    y3.name = 'y3_shuffled'
    y4.name = 'y4_shuffled'
    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))

    mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
    sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))

    mu3 = mu3.reshape((mu3.shape[0],mu3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))
    sig3 = sig3.reshape((sig3.shape[0],sig3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))

    mu4 = mu4.reshape((mu4.shape[0],mu4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))
    sig4 = sig4.reshape((sig4.shape[0],sig4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))


    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'

    inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
    nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)

    inner3 = -0.5 * T.sum(T.sqr(y3 - mu3) / sig3**2 + 2 * T.log(sig3) + T.log(2 * np.pi), axis=1)
    nll3 = -logsumexp(T.log(coeff3) + inner3, axis=1)

    inner4 = -0.5 * T.sum(T.sqr(y4 - mu4) / sig4**2 + 2 * T.log(sig4) + T.log(2 * np.pi), axis=1)
    nll4 = -logsumexp(T.log(coeff4) + inner4, axis=1)

    nll = nll1 + nll2 + nll3 + nll4
    return nll

def GMMdisag5(y, mu, sig, coeff, mu2, sig2, coeff2, mu3, sig3, coeff3, mu4, sig4, coeff4, mu5, sig5, coeff5):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y1 = y[:,0].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,0,:]
    y2 = y[:,1].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,1,:]
    y3 = y[:,2].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
    y4 = y[:,3].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
    y5 = y[:,4].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')

    y1.name = 'y1_shuffled'
    y2.name = 'y2_shuffled'
    y3.name = 'y3_shuffled'
    y4.name = 'y4_shuffled'
    y5.name = 'y5_shuffled'
    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))

    mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
    sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))

    mu3 = mu3.reshape((mu3.shape[0],mu3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))
    sig3 = sig3.reshape((sig3.shape[0],sig3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))

    mu4 = mu4.reshape((mu4.shape[0],mu4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))
    sig4 = sig4.reshape((sig4.shape[0],sig4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))

    mu5 = mu5.reshape((mu5.shape[0],mu5.shape[1]//coeff5.shape[-1],coeff5.shape[-1]))
    sig5 = sig5.reshape((sig5.shape[0],sig5.shape[1]//coeff5.shape[-1],coeff5.shape[-1]))

    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'

    inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
    nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)

    inner3 = -0.5 * T.sum(T.sqr(y3 - mu3) / sig3**2 + 2 * T.log(sig3) + T.log(2 * np.pi), axis=1)
    nll3 = -logsumexp(T.log(coeff3) + inner3, axis=1)

    inner4 = -0.5 * T.sum(T.sqr(y4 - mu4) / sig4**2 + 2 * T.log(sig4) + T.log(2 * np.pi), axis=1)
    nll4 = -logsumexp(T.log(coeff4) + inner4, axis=1)

    inner5 = -0.5 * T.sum(T.sqr(y5 - mu5) / sig5**2 + 2 * T.log(sig5) + T.log(2 * np.pi), axis=1)
    nll5 = -logsumexp(T.log(coeff5) + inner5, axis=1)

    nll = nll1 + nll2 + nll3 + nll4 + nll5
    return nll

#def GMMdisag8(y, mu, sig, coeff, mu2, sig2, coeff2, mu3, sig3, coeff3, mu4, sig4, coeff4, 
#                mu5, sig5, coeff5, mu6, sig6, coeff6, mu7, sig7, coeff7, mu8, sig8, coeff8):
def GMMdisagMulti(dim, y, mu, sig, coeff, *args):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """

    y1 = y[:,0].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,0,:]
    y1.name = 'y1_shuffled'
    mu = mu.reshape((mu.shape[0],mu.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],sig.shape[1]//coeff.shape[-1],coeff.shape[-1]))
    inner1 = -0.5 * T.sum(T.sqr(y1 - mu) / sig**2 + 2 * T.log(sig) + T.log(2 * np.pi), axis=1)
    inner1.name = 'inner'
    nll1 = -logsumexp(T.log(coeff) + inner1, axis=1)
    nll1.name = 'logsum'
    nll = nll1

    if (dim>1):
        mu2, sig2, coeff2 = args[0], args[1], args[2]
        y2 = y[:,1].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')#[:,1,:]
        y2.name = 'y2_shuffled'
        mu2 = mu2.reshape((mu2.shape[0],mu2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
        sig2 = sig2.reshape((sig2.shape[0],sig2.shape[1]//coeff2.shape[-1],coeff2.shape[-1]))
        inner2 = -0.5 * T.sum(T.sqr(y2 - mu2) / sig2**2 + 2 * T.log(sig2) + T.log(2 * np.pi), axis=1)
        nll2 = -logsumexp(T.log(coeff2) + inner2, axis=1)
        nll = nll + nll2
    if (dim>2):
        mu3, sig3, coeff3 = args[3], args[4], args[5]
        y3 = y[:,2].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
        y3.name = 'y3_shuffled'
        mu3 = mu3.reshape((mu3.shape[0],mu3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))
        sig3 = sig3.reshape((sig3.shape[0],sig3.shape[1]//coeff3.shape[-1],coeff3.shape[-1]))
        inner3 = -0.5 * T.sum(T.sqr(y3 - mu3) / sig3**2 + 2 * T.log(sig3) + T.log(2 * np.pi), axis=1)
        nll3 = -logsumexp(T.log(coeff3) + inner3, axis=1)
        nll = nll + nll3
    if (dim>3):
        mu4, sig4, coeff4 = args[6], args[7], args[8]
        y4 = y[:,3].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
        y4.name = 'y4_shuffled'
        mu4 = mu4.reshape((mu4.shape[0],mu4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))
        sig4 = sig4.reshape((sig4.shape[0],sig4.shape[1]//coeff4.shape[-1],coeff4.shape[-1]))
        inner4 = -0.5 * T.sum(T.sqr(y4 - mu4) / sig4**2 + 2 * T.log(sig4) + T.log(2 * np.pi), axis=1)
        nll4 = -logsumexp(T.log(coeff4) + inner4, axis=1)
        nll = nll + nll4
    if (dim>4):
        mu5, sig5, coeff5 = args[9], args[10], args[11]
        y5 = y[:,4].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
        y5.name = 'y5_shuffled'
        mu5 = mu5.reshape((mu5.shape[0],mu5.shape[1]//coeff5.shape[-1],coeff5.shape[-1]))
        sig5 = sig5.reshape((sig5.shape[0],sig5.shape[1]//coeff5.shape[-1],coeff5.shape[-1]))
        inner5 = -0.5 * T.sum(T.sqr(y5 - mu5) / sig5**2 + 2 * T.log(sig5) + T.log(2 * np.pi), axis=1)
        nll5 = -logsumexp(T.log(coeff5) + inner5, axis=1)
        nll = nll + nll5
    if (dim>5):
        mu6, sig6, coeff6 = args[12], args[13], args[14]
        y6 = y[:,5].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
        y6.name = 'y6_shuffled'
        mu6 = mu6.reshape((mu6.shape[0],mu6.shape[1]//coeff6.shape[-1],coeff6.shape[-1]))
        sig6 = sig6.reshape((sig6.shape[0],sig6.shape[1]//coeff6.shape[-1],coeff6.shape[-1]))
        inner6 = -0.5 * T.sum(T.sqr(y6 - mu6) / sig6**2 + 2 * T.log(sig6) + T.log(2 * np.pi), axis=1)
        nll6 = -logsumexp(T.log(coeff6) + inner6, axis=1)
        nll = nll + nll6
    if (dim>6):
        mu7, sig7, coeff7 = args[15], args[16], args[17]
        y7 = y[:,6].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
        y7.name = 'y7_shuffled'
        mu7 = mu7.reshape((mu7.shape[0],mu7.shape[1]//coeff7.shape[-1],coeff7.shape[-1]))
        sig7 = sig7.reshape((sig7.shape[0],sig7.shape[1]//coeff7.shape[-1],coeff7.shape[-1]))
        inner7 = -0.5 * T.sum(T.sqr(y7 - mu7) / sig7**2 + 2 * T.log(sig7) + T.log(2 * np.pi), axis=1)
        nll7 = -logsumexp(T.log(coeff7) + inner7, axis=1)
        nll = nll + nll7
    if (dim>7):
        mu8, sig8, coeff8 = args[18], args[19], args[20]
        y8 = y[:,7].dimshuffle(0, 'x').dimshuffle(0, 1, 'x')
        y8.name = 'y8_shuffled'
        mu8 = mu8.reshape((mu8.shape[0],mu8.shape[1]//coeff8.shape[-1],coeff8.shape[-1]))
        sig8 = sig8.reshape((sig8.shape[0],sig8.shape[1]//coeff8.shape[-1],coeff8.shape[-1]))
        inner8 = -0.5 * T.sum(T.sqr(y8 - mu8) / sig8**2 + 2 * T.log(sig8) + T.log(2 * np.pi), axis=1)
        nll8 = -logsumexp(T.log(coeff8) + inner8, axis=1)
        nll = nll + nll8
    #coeff = coeff.reshape((coeff.shape[0], 1,coeff.shape[1] ))
    return nll

def BiGauss(y, mu, sig, corr, binary):#x_in, theta_mu_in, theta_sig_in, corr_in, binary_in
    """
    Gaussian mixture model negative log-likelihood
    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    """
    mu_1 = mu[:, 0].reshape((-1, 1))
    mu_2 = mu[:, 1].reshape((-1, 1))

    sig_1 = sig[:, 0].reshape((-1, 1))
    sig_2 = sig[:, 1].reshape((-1, 1))

    y0 = y[:, 0].reshape((-1, 1))
    y1 = y[:, 1].reshape((-1, 1))
    y2 = y[:, 2].reshape((-1, 1))
    corr = corr.reshape((-1, 1))

    c_b =  T.sum(T.xlogx.xlogy0(y0, binary) +
                T.xlogx.xlogy0(1 - y0, 1 - binary), axis=1)

    inner1 =  ((0.5*T.log(1-corr**2)) +
               T.log(sig_1) + T.log(sig_2) + T.log(2 * np.pi))

    z = (((y1 - mu_1) / sig_1)**2 + ((y2 - mu_2) / sig_2)**2 -
         (2. * (corr * (y1 - mu_1) * (y2 - mu_2)) / (sig_1 * sig_2)))

    inner2 = 0.5 * (1. / (1. - corr**2))
    cost = - (inner1 + (inner2 * z))

    nll = -T.sum(cost ,axis=1) - c_b

    return nll


def BiGMM(y, mu, sig, coeff, corr, binary):
    """
    Bivariate Gaussian mixture model negative log-likelihood
    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    corr  : FullyConnected (Tanh)
    binary: FullyConnected (Sigmoid)
    """
    y = y.dimshuffle(0, 1, 'x')

    mu = mu.reshape((mu.shape[0],
                     mu.shape[1] / coeff.shape[-1],
                     coeff.shape[-1]))

    mu_1 = mu[:, 0, :]
    mu_2 = mu[:, 1, :]

    sig = sig.reshape((sig.shape[0],
                       sig.shape[1] / coeff.shape[-1],
                       coeff.shape[-1]))

    sig_1 = sig[:, 0, :]
    sig_2 = sig[:, 1, :]

    c_b = T.sum(T.xlogx.xlogy0(y[:, 0, :], binary) +
                T.xlogx.xlogy0(1 - y[:, 0, :], 1 - binary), axis=1)

    inner1 = (0.5 * T.log(1 - corr ** 2) +
              T.log(sig_1) + T.log(sig_2) + T.log(2 * np.pi))

    z = (((y[:, 1, :] - mu_1) / sig_1)**2 + ((y[:, 2, :] - mu_2) / sig_2)**2 -
         (2. * (corr * (y[:, 1, :] - mu_1) * (y[:, 2, :] - mu_2)) / (sig_1 * sig_2)))

    inner2 = 0.5 * (1. / (1. - corr**2))
    cost = -(inner1 + (inner2 * z))

    nll = -logsumexp(T.log(coeff) + cost, axis=1) - c_b

    return nll


def KLGaussianStdGaussian(mu, sig):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and standardized Gaussian dist.

    Parameters
    ----------
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    kl = T.sum(0.5 * (-2 * T.log(sig) + mu**2 + sig**2 - 1), axis=-1)

    return kl


def KLGaussianGaussian(mu1, sig1, mu2, sig2, keep_dims=0):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.

    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    if keep_dims:
        kl = 0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                    (sig1**2 + (mu1 - mu2)**2) / sig2**2 - 1)
    else:
        kl = T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                   (sig1**2 + (mu1 - mu2)**2) /
                   sig2**2 - 1), axis=-1)

    return kl


def grbm_free_energy(v, W, X):
    """
    Gaussian restricted Boltzmann machine free energy

    Parameters
    ----------
    to do::
    """
    bias_term = 0.5*(((v - X[1])/X[2])**2).sum(axis=1)
    hidden_term = T.log(1 + T.exp(T.dot(v/X[2], W) + X[0])).sum(axis=1)
    FE = bias_term -hidden_term

    return FE
