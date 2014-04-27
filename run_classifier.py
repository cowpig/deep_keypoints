from conv import *
from nn import *
from data_manager import *

train, labels = shuffle_all(*full_trainset())
train = np.array([img.ravel() for img in train])
labels = np.array([kpm.ravel() for kpm in labels])

strain = theano.shared(train)
slabels = theano.shared(labels)

learning_rate = 0.1

img_shape = (96, 96)
batch_size = 32

x = T.matrix('x')
y = T.imatrix('y')

initial_input = x.reshape((batch_size,1) + img_shape)

layer1 = ConvLayer(inputs=initial_input,
        image_shape=(batch_size, 1) + img_shape,
        filter_shape=(100, 1, 4, 4))


layer2 = NNLayer(inputs=layer1.output,
        image_shape=(batch_size, 20, 12, 12),
        filter_shape=(50, 20, 5, 5), poolsize=(2, 2))

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (20, 32 * 4 * 4) = (20, 512)
layer2_input = layer1.output.flatten(2)

# construct a fully-connected sigmoidal layer
layer2 = HiddenLayer(rng, input=layer2_input,
                     n_in=50 * 4 * 4, n_out=500,
                     activation=T.tanh    )

# classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)


# the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y)

# create a function to compute the mistakes that are made by the model
test_model = theano.function([x, y], layer3.errors(y))

# create a list of all model parameters to be fit by gradient descent
params = layer3.params + layer2.params + layer1.params + layer0.params

# create a list of gradients for all model parameters
grads = T.grad(cost, params)

# train_model is a function that updates the model parameters by SGD
# Since this model has many parameters, it would be tedious to manually
# create an update rule for each model parameter. We thus create the updates
# dictionary by automatically looping over all (params[i],grads[i])  pairs.
updates = []
for param_i, grad_i in zip(params, grads):
    updates.append((param_i, param_i - learning_rate * grad_i))
train_model = theano.function([index], cost, updates = updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})