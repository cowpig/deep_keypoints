from nn import *
from data_manager import *
import time

##############
# PREPARE DATA

whole_set = eyes_with_mouths_set()
np.random.shuffle(whole_set)
n_train = int(len(whole_set) * 0.6)
n_valid = int(len(whole_set) * 0.15)
n_test = len(whole_set) - n_train - n_valid

xtrain = theano.shared(np.array([item[0].ravel() for item in whole_set[:n_train]]).astype("float32"), 
						borrow=True)
ytrain = theano.shared(np.array([item[1] for item in whole_set[:n_train]]).astype("int32"), 
						borrow=True)

xvalid = theano.shared(np.array(
			[item[0].ravel() for item in whole_set[n_train:n_train+n_valid]]).astype("float32"), 
		borrow=True)
yvalid = theano.shared(np.array(
			[item[1] for item in whole_set[n_train:n_train+n_valid]]).astype("int32"), 
		borrow=True)

xtest = theano.shared(np.array(
			[item[0].ravel() for item in whole_set[n_train+n_valid:]]).astype("float32"), 
		borrow=True)
ytest = theano.shared(np.array(
			[item[1] for item in whole_set[n_train+n_valid:]]).astype("int32"), 
		borrow=True)


###############
# BUILD MODEL

index = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')

batch_size = 32
learning_rate = 0.01

with open("data/save/aa/single_layer/eyes_epoch_9900_291.0580.pkl", "rb") as f:
	W, b, _ = cPickle.load(f)

layer1 = NNLayer(inputs=x, n_in=19*25, n_out=500, activation=T.nnet.sigmoid, W=W, b=b)
layer2 = NNLayer(inputs=layer1.output, n_in=500, n_out=1, activation=T.nnet.sigmoid)

L1_reg = 0.00001
L1 = abs(layer1.W).sum() + abs(layer2.W).sum()

cost = (-1.0/batch_size) * (T.dot(layer2.output.reshape(y.shape), y) +
			T.dot((1-layer2.output).reshape(y.shape), (1-y)) ) + L1_reg * L1

errors = T.mean(T.neq(T.round(layer2.output).reshape(y.shape), y))

gparams = []
for param in layer1.params + layer2.params:
	gparam = T.grad(cost, param)
	gparams.append(gparam)

updates = []
for param, gparam in zip(layer1.params + layer2.params, gparams):
	updates.append((param, param - learning_rate * gparam))


train_function = theano.function(inputs=[index], 
		outputs=cost,
		updates=updates,
		givens={
			x: xtrain[index * batch_size:(index + 1) * batch_size],
			y: ytrain[index * batch_size:(index + 1) * batch_size]})

valid_function = theano.function(inputs=[index],
		outputs=errors,
		givens={
			x: xvalid[index * batch_size:(index + 1) * batch_size],
			y: yvalid[index * batch_size:(index + 1) * batch_size]})

test_function = theano.function(inputs=[index],
		outputs=errors,
		givens={
			x: xtest[index * batch_size:(index + 1) * batch_size],
			y: ytest[index * batch_size:(index + 1) * batch_size]})


###############
# TRAIN MODEL 

print '... training'

patience = 10001

n_train_batches = len(ytrain.get_value()) / batch_size
n_valid_batches = len(yvalid.get_value()) / batch_size
n_test_batches = len(ytest.get_value()) / batch_size

min_improvement = 0.995  

best_params = None
best_valid_score = np.inf
best_iter = 0
test_score = 0.
start_time = time.clock()
train_costs = []
valid_scores = []
test_scores = []

fig = plt.figure()


def get_score(function, n_batches):
	scores = [function(i) for i in xrange(n_batches)]
	minscore = 9999999
	min_index = -1
	for i, score in enumerate(scores):
		if score < minscore:
			minscore = score
			min_index = i

	# tile_raster_images(yvalid.get_value()[min_index:min_index+32], img_shape=(19,25), 
	# 		tile_shape=(19, 25), tile_spacing=(0, 0))

	return np.mean(scores)

done = False
i = 0
epoch = 0
start_time = time.clock()
while not done:
	epoch += 1
	for batch in xrange(n_train_batches):
		minibatch_cost = train_function(batch)
		train_costs.append(minibatch_cost)

		if (i+1) % 100 == 0:
			valid_score = get_score(valid_function, n_valid_batches)
			valid_scores.append((i, valid_score))
			
			print 'Epoch {}, minibatch {}/{}, validation error:\n\t{:.05f}'.format(
										epoch, batch, n_train_batches, valid_score)

			if valid_score < best_valid_score:
				if valid_score < best_valid_score * min_improvement:
					patience += 1000

				best_valid_score = valid_score
				best_iter = i

				test_score = get_score(test_function, n_test_batches)
				test_scores.append((i, test_score))

				print "Test score:\n\t{:.05f}".format(test_score)

		i += 1
		if patience <= i:
			done = True
			break

print "completed {} epochs in {}".format(epoch, time.clock() - start_time)

# with open("data/scores/eye_class_no_pretrain.pkl", "wb") as f:
#     cPickle.dump([train_costs, valid_scores, test_scores], f)

# with open("data/scores/eye_class_with_pretrain.pkl", "rb") as f:
# 	train_aa, valid_aa, test_aa = cPickle.load(f)

# with open("data/scores/eye_class_no_pretrain.pkl", "rb") as f:
# 	train_nn, valid_nn, test_nn = cPickle.load(f)

