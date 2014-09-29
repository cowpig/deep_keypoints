import conv
import nn
import trainer
import data_manager as dm
import numpy as np
import theano
import theano.tensor as T

floatx = theano.config.floatx

train, labels = dm.shuffle_all(*dm.full_trainset())
train = np.array([img.ravel() for img in train], dtype=floatx)
labels = np.array([keymap.ravel() for keymap in labels], dtype=floatx)
n_train = len(train)

s_train = theano.shared(train, borrow=True)
s_labels = theano.shared(labels, borrow=True)

x = T.matrix('x')
y = T.matrix('y')

# given that memory is an issue, programmatically get slices of images

the_layer = nn.NNLayer(inputs=x, n_in=96*96, n_out=96*96)
cost = -T.sum(y * T.log(the_layer.output) + 
                        (1 - y) * T.log(1 - the_layer.output), axis=1)

the_trainer = trainer.Trainer([the_layer], cost, x, s_train, y, s_labels)

the_trainer.run_epochs()