import numpy as np
import theano.tensor as T
import theano
from theano.tensor.signal import downsample

class ConvLayer(object):
	def __init__(self, inputs, filter_shape, image_shape):#, poolsize=(1, 1)):
		assert image_shape[1] == filter_shape[1]
		self.inputs = inputs
		rng = np.random.RandomState(23455)

		fan_in =  np.prod(filter_shape[1:])
		W_values = np.asarray(rng.uniform(
			  low=-np.sqrt(3./fan_in),
			  high=np.sqrt(3./fan_in),
			  size=filter_shape), dtype=theano.config.floatX)

		self.W = theano.shared(value=W_values, name='W')

		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, name='b')

		# convolve input feature maps with filters
		conv_out = T.nnet.conv.conv2d(inputs, self.W,
				filter_shape=filter_shape, image_shape=image_shape)

		self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))		
		# pooled_out = downsample.max_pool_2d(input=conv_out,
		# 									ds=poolsize, ignore_border=True)

		# self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		self.params = [self.W, self.b]
