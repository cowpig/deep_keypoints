import conv
import nn
import trainer
import data_manager as dm
import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

train, labels = dm.shuffle_all(*dm.full_trainset())
train = np.array([img.ravel() for img in train], dtype=floatX)
labels = np.array([keymap.ravel() for keymap in labels], dtype=floatX)
n_train = len(train)

s_train = theano.shared(train, borrow=True)
s_labels = theano.shared(labels, borrow=True)

x = T.matrix('x')
y = T.matrix('y')

batch_size = 64
img_shape = (96,96)
dropout_pct = 0.3
initial_input = x.reshape((batch_size,1) + img_shape)

# given that memory is an issue, programmatically get slices of images

conv_layer = conv.ConvLayer(inputs=initial_input, filter_shape=(50, 1, 4, 4),
							image_shape=(batch_size, 1) + img_shape,
							activation=theano.tensor.nnet.sigmoid, poolsize=(2, 2))
framesize = 50*47*47

conv_to_nn = conv_layer.output.reshape((batch_size, framesize))

dropout  = T.shared_randomstreams.RandomStreams(1234).binomial(
			(conv_to_nn.shape), n=1, p=1-(dropout_pct), 
            dtype=theano.config.floatX)

nn_layer = nn.NNLayer(inputs=conv_to_nn, n_in=framesize, n_out=96*96, 
						activation=theano.tensor.nnet.sigmoid)

entropy = -T.sum(y * T.log(nn_layer.output) + 
                        (1 - y) * T.log(1 - nn_layer.output), axis=1)
# cast necessary because mean divides by length, an int64 
cost = T.cast(T.mean(entropy), floatX)


the_trainer = trainer.Trainer([conv_layer, nn_layer], cost, x, s_train, y, s_labels)
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
