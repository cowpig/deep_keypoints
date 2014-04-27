from conv import *
from nn import *
from data_manager import *
import cPickle
import time


train, labels = shuffle_all(*full_trainset())
train = np.array([img.ravel() for img in train])
labels = np.array([kpm.ravel() for kpm in labels])

strain = theano.shared(train)
slabels = theano.shared(labels)

learning_rate = 0.1

img_shape = (96, 96)
batch_size = 32

x = T.matrix('x')
y = T.matrix('y')

initial_input = x.reshape((batch_size,1) + img_shape)

layer1 = ConvLayer(inputs=initial_input,
        image_shape=(batch_size, 1) + img_shape,
        filter_shape=(10, 1, 4, 4))

glue = layer1.output.flatten(2)


layer2 = NNLayer(inputs=glue,
        n_in=86490, n_out=96*96, activation=T.nnet.sigmoid)

entropy = -T.sum(y * T.log(layer2.output) + 
                        (1 - y) * T.log(1 - layer2.output), axis=1)

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
        with open("data/save/conv/epoch_{}.pkl".format(run_epochs.n), "wb") as f:
        	cPickle.dump([layer1, layer2, epochs.costs])

for i in xrange(100):
	run_epochs(training_function, 100, batch_size)