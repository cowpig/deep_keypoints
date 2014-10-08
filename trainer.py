import theano
import theano.tensor as T
import numpy as np
import time
from collections import OrderedDict

def epoch(batch_size_to_use, n_train, training_function):
	i=0
	costs = []
	while i + batch_size_to_use <= n_train:
		costs.append(training_function(i, batch_size_to_use))
		i += batch_size_to_use
	return costs

class Trainer(object):
	# takes a list of NN layers, all of which will be updated with respect
	#	to the provided cost function. In the case where only one layer needs
	#	to be trained, only provide that layer to the trainer.
	def __init__(self, layers, cost, x, shared_x, y, shared_y, x_mask=None,
						y_mask=None, shared_mask=None, valid_x=None, valid_y=None,
						test_x=None, test_y=None, error_func=None, batch_size=64, 
						learning_rate=0.05, momentum=0.9, weight_decay=0.0002):

		if type(layers) == list:
			self.layers = layers
		else:
			self.layers = [layers]
		self.cost = cost
		self.x = x
		self.shared_x = shared_x

		# the target output. this can be left as None in the case of autoencoder
		#	training, and instead the input (x) will be used
		self.y = y
		self.shared_y = shared_y
		
		# mask refers to a special case where we want to multiply the cost matrix
		#	by some "mask" matrix in order to manipulate the training
		self.x_mask = x_mask
		self.shared_mask = shared_mask

		# supervised training
		self.valid_x = valid_x
		self.valid_y = valid_y
		self.test_x = test_x
		self.test_y = test_y

		self.supervised = valid_x and valid_y and test_x and test_y

		# to use if training and valid/test errors are different functions
		if not error_func == None:
			self.error_func = error_func
		else:
			self.error_func = self.cost

		# hyperparameters:
		self.batch_size = batch_size

		if theano.config.floatX == "float32":
			self.momentum = np.float32(momentum)
			self.weight_decay = np.float32(weight_decay)
			self.learning_rate = np.float32(learning_rate)
		elif theano.config.floatX == "float64":
			self.momentum = np.float_(momentum)
			self.weight_decay = np.float_(weight_decay)
			self.learning_rate = np.float_(learning_rate)

		# theano tensors to be used in training
		self.params = [param for layer in self.layers for param in layer.params]
		self.gradients = T.grad(self.cost, self.params)
		# keeps track of momentum during gradient descent
		if momentum:
			self.momentums = {
					param : theano.shared(np.zeros(
						param.get_value().shape, dtype=theano.config.floatX),
						borrow=True) 
					for param in self.params
			}

		self.epochs = 0

	def get_training_function(self):
		lr, wd, mom = T.scalars('lr', 'wd', 'mom')
		i, bs = T.iscalars('i', 'bs')

		given = {self.x : self.shared_x[i : i + bs],
				     lr : self.learning_rate,
				     wd : self.weight_decay}

		if self.momentum:
			given[mom] = self.momentum

		if self.y:
			given[self.y] = self.shared_y[i : i + bs]

		if self.x_mask:
			given[self.x_mask] = self.shared_mask[i : i + bs]

		self.updates = OrderedDict()
		for param, grad in zip(self.params, self.gradients):
			if self.momentum:
				update = mom * self.momentums[param] - lr * grad
				self.updates[self.momentums[param]] = update
			else:
				update = - lr * grad

			self.updates[param] = param * (1 - wd) + update

		# import pdb; pdb.set_trace()

		return theano.function([i, bs], self.cost, 
						updates=self.updates, 
						givens=given)

	def get_valid_test_functions(self):
		given_valid = {self.x : self.valid_x,
				self.y : self.valid_y}
		valid_func = theano.function([], self.error_func, givens=given_valid)

		given_test = {self.x : self.test_x,
				self.y : self.test_y}
		test_func = theano.function([], self.error_func, givens=given_test)


	def run_epochs(self, min_epochs=50, min_improvement=1.001, 
					lr_decay=0.1, decay_modulo=25, custom_training=None):
		start = time.time()

		if custom_training is not None:
			training_function = custom_training
			print "Training using custom training function..."
		else:
			training_function = self.get_training_function()

		# train for at least this many epochs
		epoch_stop = min_epochs
		n_train = len(self.shared_x.get_value())

		if self.supervised:
			print "Beginning supervised training..."

			valid_function, test_function = self.get_valid_test_functions()
			self.best_cost_valid = float('inf')
			best_cost_test = float('inf')

			self.train_costs = []
			self.valid_costs = [(0, float('inf'))]
			self.test_costs = [(0, float('inf'))]

			while self.epochs < epoch_stop:
				self.epochs += 1
				self.train_costs.append(epoch(self.batch_size, n_train, training_function))
				print "\tTraining cost at epoch {}: {:.5f}".format(self.epoch, self.train_costs[-1])

				if self.epochs % 5 == 0:
					validation_cost = valid_function()
					self.valid_costs.append((self.epochs, validation_cost))
					print "Validation score: {:.5f} (previous best {:.5f})".format(
							validation_cost, self.best_cost_valid)
					print "\timprovement ratio: {}".format(self.best_cost_valid/validation_cost)
					if (validation_cost * min_improvement) < self.best_cost_valid:
						# with sufficient improvement, delay stopping of training
						epoch_stop += 5

						test_cost = test_function()
						print "\tTest score: {:.5f} (prev best {:.5f})".format(
							test_cost, best_cost_test)
						print "\t\timprovement ratio: {}".format(best_cost_test/test_cost)


		else:
			print "Beginning unsupervised training..."

			since_last_decay = 0
			best_cost = float('inf')
			self.costs = [(0, float('inf'))]

			while self.epochs < epoch_stop:
				self.epochs += 1
				costs = epoch(self.batch_size, n_train, training_function)

				print "=== epoch {} ===".format(self.epochs)
				print "avg cost: {:.5f} (prev best: {:.5f}; ratio: {:.5f})".format(
								np.mean(costs), best_cost, best_cost/np.mean(costs))
				
				# keep training as long as we are improving enough
				if (np.mean(costs) * min_improvement) < best_cost:
					epoch_stop += 1
					since_last_decay += 1
				elif lr_decay and since_last_decay - decay_modulo > 0:
					self.learning_rate *= (1 - lr_decay)
					since_last_decay = 0
					print "no improvement; decreasing learning rate to {}".format(
																self.learning_rate)
					print "epochs left: {}".format(epoch_stop)

				if np.mean(costs) < best_cost:
					best_cost = np.mean(costs)

				self.costs.append((self.epochs, np.mean(costs)))


		elapsed = (time.time() - start)
		print "ELAPSED TIME: {}".format(elapsed)
