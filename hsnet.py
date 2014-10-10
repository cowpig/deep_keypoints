import conv
import nn
import trainer
import data_manager as dm
import numpy as np
import theano
import theano.tensor as T

batch_size = 64
img_shape = (96,96)
# bugcheck
# img_shape = (12,12)
conv_filter_shape = (50, 1, 5, 5)
conv_stride = (1,1)
conv_poolsize = (2,2)

floatX = theano.config.floatX
data, labels = dm.shuffle_all(*dm.full_trainset())

# bugcheck
# data = np.array([img.ravel()[:12**2] for img in data], dtype=floatX)[:320]
# labels = np.array([keymap.ravel()[:12**2] for keymap in labels], dtype=floatX)[:320]

data = np.array([img.ravel() for img in data], dtype=floatX)
labels = np.array([keymap.ravel() for keymap in labels], dtype=floatX)


n_train = int(len(data) * 0.7) - int(len(data) * 0.7) % batch_size
n_test = int(len(data) - n_train) / 2

trainx = theano.shared(data[:n_train], borrow=True)
trainy = theano.shared(labels[:n_train], borrow=True)

validx = theano.shared(data[n_train:n_train+n_test], borrow=True)
validy = theano.shared(labels[n_train:n_train+n_test], borrow=True)
testx = theano.shared(data[-n_test:], borrow=True)
testy = theano.shared(labels[-n_test:], borrow=True)

x = T.matrix('x')
y = T.matrix('y')


dropout_pct = 0.3
initial_input = x.reshape((batch_size,1) + img_shape)

# given that memory is an issue, programmatically get slices of images

conv_layer = conv.ConvLayer(inputs=initial_input, filter_shape=conv_filter_shape,
							image_shape=(batch_size, 1) + img_shape, stride=conv_stride,
							activation=theano.tensor.nnet.sigmoid, poolsize=conv_poolsize)
framesize = 50*46*46
# again for bugchecking
# framesize = 50*4*4

conv_to_nn = conv_layer.output.reshape((batch_size, framesize))

dropout  = T.shared_randomstreams.RandomStreams(1234).binomial(
			(conv_to_nn.shape), n=1, p=1-(dropout_pct), 
            dtype=theano.config.floatX)

nn_layer = nn.NNLayer(inputs=conv_to_nn, n_in=framesize, n_out=img_shape[0]*img_shape[1], 
						activation=theano.tensor.nnet.sigmoid)

entropy = -T.sum(y * T.log(nn_layer.output) + 
                        (1 - y) * T.log(1 - nn_layer.output), axis=1)
# cast necessary because mean divides by length, an int64 
cost = T.cast(T.mean(entropy), floatX)


# needed for validation/testing.. 
# TODO: I should refactor this into the conv/nnet classes
test_input = x.reshape((n_test,1) + img_shape)
convolutions = T.nnet.conv.conv2d(test_input, conv_layer.W,
				filter_shape=conv_filter_shape, image_shape=(n_test, 1) + img_shape,
				subsample=conv_stride)

pools = T.signal.downsample.max_pool_2d(input=convolutions, ds=conv_poolsize, ignore_border=False)
conv_activation = T.nnet.sigmoid(pools + conv_layer.b.dimshuffle('x', 0, 'x', 'x'))
conv_reshape = conv_activation.reshape((n_test, framesize))
connected_output = T.nnet.sigmoid(T.dot(conv_reshape, nn_layer.W) + nn_layer.b)
uncorrupted_entropy = -T.sum(y * T.log(connected_output) + 
                        (1 - y) * T.log(1 - connected_output), axis=1)
error = T.cast(T.mean(uncorrupted_entropy), floatX)

the_trainer = trainer.Trainer([conv_layer, nn_layer], cost, x, trainx, y, trainy,
							valid_x=validx, valid_y=validy, test_x=testx, test_y=testy, 
							error_func=error)
func = the_trainer.get_training_function()

the_trainer.run_epochs()

params["Wconv"] = conv_layer.w.value()
params["Wconv"] = conv_layer.W.value()
params["Wconv"] = conv_layer.W.get_value()
params["bconv"] = conv_layer.b.get_value
params["bconv"] = conv_layer.b.get_value()
params["Wnn"] = nn_layer.W.get_value()
params["bnn"] = nn_layer.b.get_value()
np.savez_compressed("conv_first_try_dropout", **params)
