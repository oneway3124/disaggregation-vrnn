##import ipdb
import logging
import theano.tensor as T
import time
import numpy as  np
from cle.cle.graph import TheanoMixin
from cle.cle.models import Model
from cle.cle.utils import PickleMixin, tolist

from collections import defaultdict
from theano.compat.python2x import OrderedDict
from math import exp
from itertools import izip


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Training(PickleMixin, TheanoMixin):
    """
    WRITEME

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 name,
                 data,
                 model,
                 optimizer,
                 cost,
                 outputs,
                 n_steps,
                 debug_print=0,
                 trainlog=None,
                 extension=None,
                 lr_iterations=None,
                 decay_schedule = 2,
                 k_speedOfconvergence = 40):
        #picklelized?
        self.name = name # yes
        self.data = data # no
        self.model = model #yes
        self.optimizer = optimizer #no
        self.inputs = model.inputs #no
        self.cost = cost #yes
        self.outputs = tolist(outputs) #no
        self.updates = OrderedDict() # no
        self.updates.update(model.updates) #???
        self.extension = extension #no
        self.debug_print = debug_print #no
        lr_scalers = OrderedDict() #yes
        for node in self.model.nodes: #should
            lr_scalers[node.name] = node.lr_scaler
        self.optimizer.lr_scalers = lr_scalers #should
        self.nBernoulli = np.ones((n_steps,)) #yes
        t0 = time.time() 
        self.cost_fn = self.build_training_graph() # no but should
        print "Elapsed compilation time: %f" % (time.time() - t0)
        if self.debug_print: #no
            from theano.printing import debugprint
            debugprint(self.cost_fn)
        if trainlog is None: #yes
            self.trainlog = TrainLog()
        else:
            self.trainlog = trainlog
        self.endloop = 0 #no
        self.lr_iterations = lr_iterations #yes
        self.lastBatchlastPoch = 0 #yes
        self.decay_schedule = decay_schedule #yes
        self.k = k_speedOfconvergence #yes 
        self.schedRate = 1 #yes
        self.n_steps = n_steps #yes

    def restore(self,
                 data,
                 optimizer,
                 cost,
                 outputs,
                 n_steps,
                 debug_print=0,
                 trainlog=None,
                 extension=None,
                 lr_iterations=None,
                 decay_schedule = 2,
                 k_speedOfconvergence = 40):
        self.data = data
        self.optimizer = optimizer
        self.inputs = self.model.inputs
        self.cost = cost
        self.outputs = tolist(outputs)
        #self.updates = OrderedDict()
        #self.updates.update(self.model.updates)
        self.updates = self.model.updates
        self.extension = extension
        self.debug_print = debug_print
        lr_scalers = OrderedDict()
        for node in self.model.nodes:
            lr_scalers[node.name] = node.lr_scaler
        self.optimizer.lr_scalers = lr_scalers
        self.nBernoulli = np.ones((n_steps,))
        t0 = time.time()
        self.cost_fn = self.build_training_graph()
        print "Elapsed compilation time: %f" % (time.time() - t0)
        if self.debug_print:
            from theano.printing import debugprint
            debugprint(self.cost_fn)
        if trainlog is None:
            self.trainlog = TrainLog()
        else:
            self.trainlog = trainlog
        self.endloop = 0
        self.lr_iterations = lr_iterations
        self.lastBatchlastPoch = 0
        self.decay_schedule = decay_schedule
        self.k = k_speedOfconvergence
        self.schedRate = 1
        self.n_steps = n_steps
    '''
    def restore(self,
                data,
                cost,
                model,
                optimizer,
                k_speedOfconvergence = 40):
        self.data = data
        self.cost = cost
        self.model = model
        self.optimizer = optimizer
        self.inputs = model.inputs
        lr_scalers = OrderedDict()
        for node in self.model.nodes:
            lr_scalers[node.name] = node.lr_scaler
        self.cost_fn = self.build_training_graph()
        self.k = k_speedOfconvergence
    '''
    def build_training_graph(self):

        self.run_extension('ext_regularize_pre_grad')
        self.grads = OrderedDict(izip(self.model.params.values(),
                                      T.grad(self.cost, self.model.params.values())))
        self.run_extension('ext_grad')
        grads = self.optimizer.get_updates(self.grads)

        for key, val in grads.items():
            self.updates[key] = val

        self.run_extension('ext_regularize_post_grad')
        print(type(self.inputs), len(self.inputs))
        #self.inputs.append(self.nBernoulli)
        return self.build_theano_graph(self.inputs, self.outputs, self.updates)

    def run(self):
        logger.info("Entering main loop")
        while self.run_epoch():
            pass
        logger.info("Terminating main loop")

    def run_epoch(self):
        self.trainlog.lastBatchlastEpoch = self.trainlog.batch_seen
        
        for batch in self.data:
            self.run_extension('ext_monitor')
            self.run_extension('ext_save')
            batch_t0 = time.time()
            nBernoulli = [np.random.binomial(1,self.schedRate) for i in range(self.n_steps)]
            nBernoulli = np.asarray(nBernoulli)
            nBernoulli = np.reshape(nBernoulli,(self.n_steps,))
            batchAux = (batch + (nBernoulli,))

            this_cost = self.cost_fn(*batchAux)
            self.trainlog.monitor['time'].append(time.time() - batch_t0)
            self.trainlog.monitor['update'].append(this_cost)
            self.trainlog.batch_seen += 1
            self.run_extension('ext_schedule')


        self.trainlog.epoch_seen += 1
        first = self.trainlog.epoch_seen/float(self.k)
        second = self.k + exp(first)
        self.schedRate = self.k/second
        for limit, lr_it in self.lr_iterations.items():
            if (limit < self.trainlog.epoch_seen):
                self.optimizer.lr.set_value(lr_it)
        print("Epoch: {} - seched rate: {}".format(self.trainlog.epoch_seen,self.schedRate))
        self.run_extension('ext_term')## changes the value of endloop

        if self.end_training():
            self.run_extension('ext_monitor')
            self.run_extension('ext_save')
            return False

        return True

    def find_extension(self, name):

        try:
            exts = [extension for extension in self.extension
                    if extension.name == name]
            if len(exts) > 0:
                return_val = 1
            else:
                return_val = 0
            return return_val, exts
        except:
            return (0, None)

    def run_extension(self, name):
        tok, exts = self.find_extension(name)
        if tok:
            for ext in exts:
                ext.exe(self)

    def end_training(self):
        return self.endloop


class TrainLog(object):
    """
    Training log class

    Parameters
    ----------
    .. todo::
    """
    def __init__(self):
        self.monitor = defaultdict(list)
        print("Initial dictionary ")
        print(self.monitor.keys())
        self.epoch_seen = 0
        self.batch_seen = 0
        self.lastBatchlastEpoch = 0
