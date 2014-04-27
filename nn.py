import numpy as np
import theano
import theano.tensor as T


class NNLayer(object):
    def __init__(self, inputs, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.inputs = inputs

        rng = numpy.random.RandomState(1234)

        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
        else:
            W_values = W

        W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        else:
            b_values = b
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(inputs, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))

        self.params = [self.W, self.b]