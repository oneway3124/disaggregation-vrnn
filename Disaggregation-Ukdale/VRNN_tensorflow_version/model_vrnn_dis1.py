import tensorflow as tf
import numpy as np

#function to define a layer in the network 
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    ''' 
	Defines a layer in the graph with input data number
	of neurons

	Arguments

	input_ : input tensor to the layer
	output_size : number of outputs from the layer
	scope : Scope/Name of the Layer 
	stddev : standard deviation of the distribution for initializer
	bias_start : constant at which initialization of bias begins
	with_w : if True return the weights and bias of the layer

	Return : the output of (weight * input) + bias ( with or without weight and bias matrices)
    '''

    #get the shape of input data
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        #define weight matrix for the layer and assign the initializer to random_normal
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))

	#define the bias variable
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))

	#perform the summation operation of a neuron and return 
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


class VartiationalRNNCell(tf.contrib.rnn.RNNCell):
    """Variational RNN cell. 

       The model for the training is constructed in this cell.
    """

    def __init__(self, h_dim, z_dim = 100, target_dim = 15):
        '''
        Initializes the VariationalRNN Cell objects when created.

	Arguments

	h_dim : output size of lstm cell
	z_dim : size of the latent variable
	target_dim : size of Decoder output
        '''

	q_z_dim = 150
	p_z_dim = 150
	p_x_dim = 200
	x2s_dim = 100
	y2s_dim = 100
	z2s_dim = 100

        self.n_h = h_dim
        self.n_x = target_dim
        self.n_z = z_dim
        self.n_x_1 = x2s_dim
	self.n_y_1 = y2s_dim
        self.n_z_1 = z2s_dim
        self.n_enc_hidden = p_z_dim
        self.n_dec_hidden = p_x_dim
        self.n_prior_hidden = p_z_dim
        self.lstm = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True) #define the LSTM cell


    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, x, state, scope=None):

	'''
	This method is called when the object of the class is used as a function call.

	Arguments

	x : input data to the model, it is a tuple of (x, y)
	state : initial state of the model
	scope : scope or name under which the model is defined in the graph

	Returns : The outputs of encoder layers, decoder layers, RNN layer and the state of the RNN layer as a tuple 
	
	'''

        def GMM_sample(mu,sigma,coeff):

		'''
		Performs the gmm sampling on the decoder's output

		Arguments

		mu : The mean of the distributions representing each output
		sigma : The standard deviation of the distributions representing each output
		coeff : the weights for each of the distributions

		Returns: the output values each sampled from their respective distributions
		'''
		#find the index of the distribution with maximum weight for each timestep
		idx = tf.argmax(coeff,axis =1)
		
		#indicies of mu and sigma of the distribution with max weight for each timestep 
		indicies = [[x,idx[x]] for x in range(idx.get_shape()[0])]

		#collecting only the required values from the decoder(theta) output
		mu = tf.gather_nd(mu,indicies)
		sigma = tf.gather_nd(sigma,indicies)
		
		#random value from a standard normal distribution
		epsilon = tf.random_normal(mu.shape,dtype=tf.float32)

		#return the output of the model for this timestep
		return tf.reshape(mu + sigma * epsilon,[250,1])

        #define the model under the scope or the class name
        with tf.variable_scope(scope or type(self).__name__):
            h, c = state #state of lstm cell

            #define the x_1 layer with relu activation
            with tf.variable_scope("x_1"):
                x_1 = tf.nn.relu(linear(x[0], self.n_x_1), name='x_1')

            #define the y_1 layer with relu activation
            with tf.variable_scope("y_1"):
                y_1 = tf.nn.relu(linear(x[1], self.n_y_1), name='y_1')

            with tf.variable_scope("Prior"):
                #define the prior_1 with relu activation
                with tf.variable_scope("hidden"):
                    prior_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(x_1,h)), self.n_prior_hidden), name='prior_1')
                #define the prior_mu 
                with tf.variable_scope("mu"):
                    prior_mu = linear(prior_hidden, self.n_z)
                #define the prior_sig with softplus activation
                with tf.variable_scope("sigma"):
                    prior_sigma = tf.nn.softplus(linear(prior_hidden, self.n_z))

            with tf.variable_scope("Phi"):
                #define the phi_1 with relu activation
                with tf.variable_scope("hidden"):
                    phi_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(x_1, y_1, h)), self.n_enc_hidden))
                #define the phi_mu
                with tf.variable_scope("mu"):
                    phi_mu    = linear(phi_hidden, self.n_z)
                #define the phi_sig with softplus activation
                with tf.variable_scope("sigma"):
                    phi_sigma = tf.nn.softplus(linear(phi_hidden, self.n_z))

            #Reparametrization Trick / Gaussian Sampling 
            eps = tf.random_normal((x[0].get_shape().as_list()[0], self.n_z), 0.0, 1.0, dtype=tf.float32)
            # z = mu + sigma*epsilon
            z = tf.add(phi_mu, tf.multiply(phi_sigma, eps))

            #define the z_1 with relu activation
            with tf.variable_scope("z_1"):
                z_1 = tf.nn.relu(linear(z, self.n_z_1))

            with tf.variable_scope("Theta"):
                #define the theta_1 with relu activation
                with tf.variable_scope("hidden"):
                    theta_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(z_1, h)), self.n_dec_hidden))
                #define the theta_mu
                with tf.variable_scope("mu"):
                    theta_mu = linear(theta_hidden, self.n_x)
                #define the theta_sig with softplus activation
                with tf.variable_scope("sigma"):
                    theta_sigma = tf.nn.softplus(linear(theta_hidden, self.n_x))
                #define the theta_coeff with softmax activation
                with tf.variable_scope("coeff"):
                    theta_coeff = tf.nn.softmax(linear(theta_hidden, self.n_x))

            #GMM sampling to obtain pred
            pred = GMM_sample(theta_mu,theta_sigma,theta_coeff)

            #define the RNN layer (uses the LSTM cell defined in init)
            output2, state2 = self.lstm(tf.concat(axis=1,values=(x_1, y_1, z_1)), state)

        return (phi_mu, phi_sigma, theta_mu, theta_sigma, theta_coeff, prior_mu, prior_sigma, pred), state2




class VRNN():
    def __init__(self, args):
	'''
	This class handles the creating the model object and the backpropagation over the layers.

	Arguments
	args : the configuration arguments of the dataset

	'''
        def tf_normal(y, mu, s, rho):
	    '''
	    This method find the negative log likelihood loss.

	    Arguments
	    y : The prediction or output of the model
	    mu : The mean of the distribution of y (theta_mu)
	    sigma : The stddev of the distribution of y (theta_sig)
	    rho : The coefficients of the distribution of y (theta_coeff)
	    '''
            with tf.variable_scope('normal'):
                ss = tf.maximum(1e-10,tf.square(s))
                norm = tf.subtract(y[:,:args.chunk_samples], mu) 
		#broadcasting - tensorflow can broadcast the dimension if the last dimesion is 1,
		#thus here even though the dimensions are (50000,1) and (50000,15) subtract works
                z = tf.div(tf.square(norm), ss)
                denom_log = tf.log(2*np.pi*ss, name='denom_log')
                result = tf.reduce_sum(z+denom_log, 1)/2
            return result

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
	    '''
	     This method finds the kl_dovergence loss which signifies how two distributions are different from each other.

	    Arguments
	    mu_1, mu_2 : Mean of distributions
	    sigma_1, sigma_2 : Standard Deviation of distributions
	    
	    '''
            with tf.variable_scope("kl_gaussgauss"):
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-9,sigma_2),name='log_sigma_2') 
                  - 2 * tf.log(tf.maximum(1e-9,sigma_1),name='log_sigma_1')
                  + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9,(tf.square(sigma_2))) - 1
                ), 1)

        def get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma, y):
	    '''This method computes the loss value of the model for a batch of input.

	    Arguments:
		To be done

	    Returns: The loss value
	    '''
	    #find the kl_divergence loss
            kl_loss = tf_kl_gaussgauss(enc_mu, enc_sigma, prior_mu, prior_sigma)

	    #find the negative log likelihood loss
            likelihood_loss = tf_normal(y, dec_mu, dec_sigma, dec_rho)

            return tf.reduce_mean(kl_loss + likelihood_loss)
       
        def shape_inp(input_):
		''' 
		This function applies transpose and reshapes the data suitable to be given to the model.

		Arguments
		input : a batch of the training data to be shaped.

		Returns : data in the shape timesteps*[batch_size,n_app]
		'''
		input_ = tf.transpose(input_, [1, 0, 2])  # permute n_steps and batch_size
		input_ = tf.reshape(input_, [-1, args.chunk_samples]) # (n_steps*batch_size, n_input)
            	# Split data because rnn cell needs a list of inputs for the RNN inner loop
		input_ = tf.split(axis=0, num_or_size_splits=args.seq_length, value=input_) # n_steps * (batch_size, n_hidden)   
		return input_

        self.args = args

	#create the object for VRNN model
        cell = VartiationalRNNCell(args.rnn_size, args.latent_size,args.target_dim)

        self.cell = cell

	#placeholders are variables that are assigned data later
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, args.chunk_samples], name='input_x') #x input data
	self.input_y = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, args.chunk_samples], name='input_y') #y input data

        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, args.chunk_samples],name = 'target_data') #target or label data (same as y here)

	#The initial states of the LSTM cell
	'''The cell.zero_state() is a function defined in tf.contrib.rnn.RNNCell (the parent class of VariationalRNNCell'''
        self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
	       #transpose and reshape input data
               input_x = shape_inp(self.input_x)
               input_y = shape_inp(self.input_y)
	       x_y_inp = (input_x,input_y)
	       #arrange input to pass into the static_rnn
	       inputs = []
               for x in range(len(input_x)):
	            inputs.append([input_x[x],input_y[x]])               

	#target data in [batch_size*timesteps,n_app] shape
        flat_target_data = tf.reshape(self.target_data,[-1, args.chunk_samples])

        self.target = flat_target_data

	#store the input in [batch_size*timesteps,1] shape
        self.flat_input = []

	self.input = []
	for input_i in x_y_inp:
               self.flat_input.append(tf.reshape(tf.transpose(tf.stack(input_i),[1,0,2]),[args.batch_size*args.seq_length, -1]))
               self.input.append(tf.stack(input_i))

        # Get vrnn cell output
        outputs, last_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=(self.initial_state_c,self.initial_state_h))
        
	#separate each of the ouput tensors listed in 'names' as separate list
        outputs_reshape = []
        names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "dec_coeff", "prior_mu", "prior_sigma","gmm_sample"]
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])
                x = tf.transpose(x,[1,0,2]) #change back axis 0 and 1
                x = tf.reshape(x,[args.batch_size*args.seq_length, -1])
                outputs_reshape.append(x)
        
        enc_mu, enc_sigma, dec_mu, dec_sigma, dec_coeff, prior_mu, prior_sigma, outputs = outputs_reshape
	
	#final outputs of the model for a batch of input        
	self.final_state_c,self.final_state_h = last_state
        self.mu = dec_mu
        self.sigma = dec_sigma
        self.coeff = dec_coeff
        self.output = outputs

	#get the loss value (combination of kl_divergence and negative log likelihood)
        lossfunc = get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_coeff, prior_mu, prior_sigma, flat_target_data)
        
        with tf.variable_scope('cost'):
            self.cost = lossfunc 

	#write into logs the cost mu and sigma
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('mu', tf.reduce_mean(self.mu))
        tf.summary.scalar('sigma', tf.reduce_mean(self.sigma))


	#agjust the weights of the layers by back propagation 
        self.lr = tf.Variable(0.0, trainable=False) #variable for learning rate

	#get all the trainable variables
        tvars = tf.trainable_variables()
        for t in tvars:
            print t.name
	
	#find the gradients for each trainable variables
        grads = tf.gradients(self.cost, tvars)
        
	#using the adam optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)

	#apply gradient descent to all layers
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
