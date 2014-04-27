import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import scale
import time

class RBM(object):
	def __init__(self, inputs=None, n_visible=19*26, n_hidden=500, W=None, hbias=None, 
				vbias=None, numpy_rng=None, theano_rng=None):

	self.n_visible = n_visible
	self.n_hidden = n_hidden


	if numpy_rng is None:
		# create a number generator
		numpy_rng = numpy.random.RandomState(1234)

	if theano_rng is None:
	ano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

	if not W:
		initial_W = numpy.asarray(numpy.random.uniform(
						low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
						high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
						size=(n_visible, n_hidden)),
						dtype=theano.config.floatX)
	else:
		initial_W = W

	self.W = theano.shared(value=initial_W, name='W')

	if not hbias:
		# create shared variable for hidden units bias
		self.hbias = theano.shared(value=numpy.zeros(n_hidden,
		dtype=theano.config.floatX), name='hbias')
	else:
		self.hbias = theano.shared(hbias, name='hbias')

	if not vbias:
		# create shared variable for visible units bias
		self.vbias = theano.shared(value =numpy.zeros(n_visible,
		dtype = theano.config.floatX),name='vbias')
	else:
		self.vbias = theano.shared(vbias, name='vbias')


	self.input = inputs if inputs else T.dmatrix('input')

	self.theano_rng = theano_rng

	self.params = [self.W, self.hbias, self.vbias]


	def sample_h_given_v(self, vis):
		pre_sigmoid_h1 = T.dot(vis, self.W) + self.hbias
		h1_mean = T.nnet.sigmoid(pre_sigmoid_h1)
		h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean,
											 dtype=theano.config.floatX)
		return [pre_sigmoid_h1, h1_mean, h1_sample]

	def sample_v_given_h(self, hid):
		pre_sigmoid_v1 = T.dot(hid, self.W.T) + self.vbias
		v1_mean = T.nnet.sigmoid(pre_sigmoid_activation)
		v1_sample = self.theano_rng.binomial(size=v1_mean.shape,n=1, p=v1_mean,
											 dtype=theano.config.floatX)
		return [pre_sigmoid_v1, v1_mean, v1_sample]


	def gibbs_v_given_h(self, h0_sample):
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

	def gibbs_h_given_v(self, v0_sample):
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]


	def free_energy(self, v_sample):
		wx_b = T.dot(v_sample, self.W) + self.hbias
		vbias_term = T.dot(v_sample, self.vbias)
		hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
		return -hidden_term - vbias_term

	def get_cost_updates(self, lr=0.1, persistent=None, k=1):
		# compute positive phase
		pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

		# decide how to initialize persistent chain:
		# for CD, we use the newly generate hidden sample
		# for PCD, we initialize from the old state of the chain
		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent

		# perform actual negative phase
		# in order to implement CD-k/PCD-k we need to scan over the
		# function that implements one gibbs step k times.
		# Read Theano tutorial on scan for more information :
		# http://deeplearning.net/software/theano/library/scan.html
		# the scan will return the entire Gibbs chain
		[pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], \
			updates = theano.scan(self.gibbs_hvh,
					# the None are place holders, saying that
					# chain_start is the initial state corresponding to the
					# 6th output
					outputs_info=[None, None, None, None, None, chain_start],
					n_steps=k)

		# determine gradients on RBM parameters
		# note that we only need the sample at the end of the chain
		chain_end = nv_samples[-1]

		cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
		# We must not compute the gradient through the gibbs sampling
		gparams = T.grad(cost, self.params, consider_constant=[chain_end])

		# constructs the update dictionary
		for gparam, param in zip(gparams, self.params):
			# make sure that the learning rate is of the right dtype
			updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)
		if persistent:
			# Note that this works only if persistent is a shared variable
			updates[persistent] = nh_samples[-1]
			# pseudo-likelihood is a better proxy for PCD
			monitoring_cost = self.get_pseudo_likelihood_cost(updates)
		else:
			# reconstruction cross-entropy is a better proxy for CD
			monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

		return monitoring_cost, updates