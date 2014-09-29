import conv
import nn
import data_manager as dm
import cPickle
import time
import theano.tensor as T
import theano
import numpy as np

train, labels = dm.shuffle_all(*dm.full_trainset())
train = np.array([img.ravel() for img in train])[:256]
labels = np.array([kpm.ravel() for kpm in labels])[:256]    
n_train = len(train)

strain = theano.shared(train)
slabels = theano.shared(labels)

learning_rate = 0.05

img_shape = (96, 96)
batch_size = 128
dropout_ratio = 0.3

x = T.matrix('x')
y = T.matrix('y')

initial_input = x.reshape((batch_size,1) + img_shape)

layer1 = conv.ConvLayer(inputs=initial_input,
        image_shape=(batch_size, 1) + img_shape,
        filter_shape=(50, 1, 4, 4))

print (batch_size, 1) + img_shape

glue = layer1.output.flatten(2)


noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                            (glue.shape), n=1, p=1-(dropout_ratio), 
                            dtype=theano.config.floatX)
dropout = noise * glue

layer2 = nn.NNLayer(inputs=dropout,
        n_in=105800, n_out=1024, activation=T.nnet.sigmoid)

layer3 = nn.NNLayer(inputs=layer2.output, n_in=1024, n_out=96*96, 
                    activation=T.nnet.sigmoid)

entropy = -T.sum(y * T.log(layer3.output) + 
                        (1 - y) * T.log(1 - layer3.output), axis=1)

cost = T.mean(entropy)

params = layer2.params + layer1.params

grads = T.grad(cost, params)

index = T.iscalar('index')

updates = []
for param_i, grad_i in zip(params, grads):
    updates.append((param_i, param_i - learning_rate * grad_i))
training_function = theano.function([index], cost, updates = updates,
        givens={
            x: strain[index * batch_size: (index + 1) * batch_size],
            y: slabels[index * batch_size: (index + 1) * batch_size]})

def epoch(n_train, training_function, batch_size):
    i=0
    costs = []
    while i*batch_size < n_train-batch_size:
        costs.append(training_function(i))
        i += 1

    return costs

def run_epochs(training_function, n_epochs, batch_size, save=True):
    if 'n' not in dir(run_epochs):
        run_epochs.n = 0

    if 'costs' not in dir(run_epochs):
        run_epochs.costs = [(0, 999999)]

    start = time.time()
    print "start time: {}".format(time)
    for x in xrange(n_epochs):
        run_epochs.n += 1
        costs = epoch(n_train, training_function, batch_size)
        print "=== epoch {} ===".format(run_epochs.n)
        print "costs: {}".format([line[()] for line in costs])
        print "avg: {}".format(np.mean(costs))
        run_epochs.costs.append((run_epochs.n, np.mean(costs)))

    elapsed = (time.time() - start)
    print "ELAPSED TIME: {}".format(elapsed)

    if save:
        with open("data/save/conv/qs/epoch_{}.pkl".format(run_epochs.n), "wb") as f:
        	cPickle.dump([layer1, layer2, epochs.costs])

for i in xrange(100):
	run_epochs(training_function, 20, batch_size)