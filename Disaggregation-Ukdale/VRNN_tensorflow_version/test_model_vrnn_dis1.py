import tensorflow as tf
import numpy as np

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

class VartiationalRNNCell(tf.contrib.rnn.RNNCell):
    """Variational RNN cell."""

    def __init__(self, x_dim, h_dim, z_dim = 100, target_dim = 15):

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
        self.lstm = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)


    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, x, state, scope=None):

        def GMM_sample(mu,sigma,coeff):
		idx = tf.argmax(coeff,axis =1)
		indicies = [[x,idx[x]] for x in range(idx.get_shape()[0])]
		mu = tf.gather_nd(mu,indicies)
		sigma = tf.gather_nd(sigma,indicies)
		epsilon = tf.random_normal(mu.shape,dtype=tf.float32)
		return tf.reshape(mu + sigma * epsilon,[250,1])

        with tf.variable_scope(scope or type(self).__name__):
            h, c = state

            with tf.variable_scope("phi_x"):
                x_1 = tf.nn.relu(linear(x, self.n_x_1))

            with tf.variable_scope("Prior"):
                with tf.variable_scope("hidden"):
                    prior_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(x_1,h)), self.n_prior_hidden))
                with tf.variable_scope("mu"):
                    prior_mu = linear(prior_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    prior_sigma = tf.nn.softplus(linear(prior_hidden, self.n_z))
            eps = tf.random_normal((x.get_shape().as_list()[0], self.n_z), 0.0, 1.0, dtype=tf.float32)
            # z = mu + sigma*epsilon
            z = tf.add(prior_mu, tf.multiply(prior_sigma, eps))
            with tf.variable_scope("phi_z"):
                z_1 = tf.nn.relu(linear(z, self.n_z_1))

            with tf.variable_scope("Decoder"):
                with tf.variable_scope("hidden"):
                    dec_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(z_1, h)), self.n_dec_hidden))
                with tf.variable_scope("mu"):
                    dec_mu = linear(dec_hidden, self.n_x)
                with tf.variable_scope("sigma"):
                    dec_sigma = tf.nn.softplus(linear(dec_hidden, self.n_x))
                with tf.variable_scope("coeff"):
                    dec_coeff = tf.nn.softmax(linear(dec_hidden, self.n_x))

            output = GMM_sample(dec_mu,dec_sigma,dec_coeff)
	    
	    with tf.variable_scope("phi_y"):
                y_1 = tf.nn.relu(linear(output, self.n_y_1), name='y_1')

            output2, state2 = self.lstm(tf.concat(axis=1,values=(x_1, y_1, z_1)), state)
        return (dec_mu, dec_sigma, dec_coeff, prior_mu, prior_sigma, output), state2




class test_VRNN():
    def __init__(self, args):

        def shape_inp(input_x):
		input_x = tf.transpose(input_x, [1, 0, 2])  # permute n_steps and batch_size
		input_x = tf.reshape(input_x, [-1, args.chunk_samples]) # (n_steps*batch_size, n_input)
            	# Split data because rnn cell needs a list of inputs for the RNN inner loop
		input_x = tf.split(axis=0, num_or_size_splits=args.seq_length, value=input_x) # n_steps * (batch_size, n_hidden)   
		return input_x

        self.args = args

        cell = VartiationalRNNCell(args.chunk_samples, args.rnn_size, args.latent_size,args.target_dim)

        self.cell = cell

        self.input_x = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, args.chunk_samples], name='input_x')
        self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
               inputs = shape_inp(self.input_x)

        self.flat_input = tf.reshape(tf.transpose(tf.stack(inputs),[1,0,2]),[args.batch_size*args.seq_length, -1])
        self.input = tf.stack(inputs)

        # Get vrnn cell output
        outputs, last_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=(self.initial_state_c,self.initial_state_h))

        outputs_reshape = []
        names = ["dec_mu", "dec_sigma", "dec_coeff", "prior_mu", "prior_sigma","gmm_sample"]
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])
                x = tf.transpose(x,[1,0,2])
                x = tf.reshape(x,[args.batch_size*args.seq_length, -1])
                outputs_reshape.append(x)
        
        dec_mu, dec_sigma, dec_coeff, prior_mu, prior_sigma, y_pred = outputs_reshape
	
        self.final_state_c,self.final_state_h = last_state
        self.mu = dec_mu
        self.sigma = dec_sigma
        self.coeff = dec_coeff

	self.pred = y_pred
        
        tf.summary.scalar('mu', tf.reduce_mean(self.mu))
        tf.summary.scalar('sigma', tf.reduce_mean(self.sigma))

        tvars = tf.trainable_variables()

	print("\nIn Test Model")
        for t in tvars:
            print t.name


