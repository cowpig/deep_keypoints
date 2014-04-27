import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import scale
import time
import matplotlib.pyplot as plt
from PIL import Image
import cPickle
import copy


class Autoencoder(object):
    def __init__(self, input_tensor, n_in, n_hidden, learning_rate, pct_blackout=0.2, 
                    W=None, b_in=None, b_out=None):
        if W == None:
            # initialization of weights as suggested in theano tutorials
            W = np.asarray(np.random.uniform(
                                        low=-4 * np.sqrt(6. / (n_hidden + n_in)),
                                        high=4 * np.sqrt(6. / (n_hidden + n_in)),
                                        size=(n_in, n_hidden)), 
                                        dtype=theano.config.floatX)

        self.W = theano.shared(W, 'W')

        if b_in == None:
            self.b_in = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'b_in')
        else:
            self.b_in = theano.shared(b_in, 'b_in')

        if b_out == None:
            self.b_out = theano.shared(np.zeros(n_in, dtype=theano.config.floatX), 'b_out')
        else:
            self.b_out = theano.shared(b_out, 'b_out')

        matrixType = T.TensorType(theano.config.floatX, (False,)*2)


        self.n_in = n_in
        self.n_hidden = n_hidden
        self.inputs = input_tensor
        self.x = matrixType('x')
        self.pct_blackout = pct_blackout
        self.noise = T.shared_randomstreams.RandomStreams(1234).binomial(
                            (self.x.shape), n=1, p=1-(self.pct_blackout), 
                            dtype=theano.config.floatX)
        self.noisy = self.noise * self.x
        self.active_hidden = T.nnet.sigmoid(T.dot(self.noisy, self.W) + self.b_in)
        self.output = T.nnet.sigmoid(T.dot(self.active_hidden, self.W.T) + self.b_out)

        self.entropy = -T.sum(self.x * T.log(self.output) + 
                                (1 - self.x) * T.log(1 - self.output), axis=1)

        self.cost = T.mean(self.entropy)

        self.parameters = [self.W, self.b_in, self.b_out]
        self.gradients = T.grad(self.cost, self.parameters)

        self.learning_rate = learning_rate

        self.updates = []
        for param, grad in zip(self.parameters, self.gradients):
            self.updates.append((param, param - self.learning_rate * grad))

        i, batch_size = T.iscalars('i', 'batch_size')
        self.train_step = theano.function([i, batch_size], self.cost, 
                                            updates=self.updates, 
                                            givens={self.x:self.inputs[i:i+batch_size]})
                                            #, mode="DebugMode")

    def copy(self, new_learning_rate=None):
        if new_learning_rate == None:
            new_learning_rate = self.learning_rate
        return Autoencoder(self.inputs, self.n_in, self.n_hidden, new_learning_rate,
                            self.pct_blackout, self.W.get_value(), self.b_in.get_value(),
                            self.b_out.get_value())

    def visualize_hidden(self, img_shape=(19,25), vis_shape=(25,20)):
        img_h, img_w = img_shape
        n_imgs_vert, n_imgs_hor = vis_shape

        assert n_imgs_hor * n_imgs_vert == self.n_hidden

        return data_manager.tile_raster_images(
            X=self.W.get_value(borrow=True).T,
            img_shape=img_shape, tile_shape=vis_shape,
            tile_spacing=(1, 1))

    def save(self, f):
        with open(f, "wb") as f:
            cPickle.dump([thing.get_value() for thing in self.parameters], f)

    def recreate(self, img, noise=False):
        img_copy = copy.deepcopy(img)

        if "recreation_function" not in dir(self):
            matrixType = T.TensorType(theano.config.floatX, (False,)*2)
            start = matrixType('imgs')
            activation = T.nnet.sigmoid(T.dot(start, self.W) + self.b_in)
            output = T.nnet.sigmoid(T.dot(activation, self.W.T) + self.b_out)
            self.recreation_function = theano.function([start], output)

        return self.recreation_function(img)

def epoch(batch_size_to_use, n_train, theano_function):
    i=0
    costs = []
    while i + batch_size_to_use < n_train:
        # print "i {}, batch_size {}, n_train {}".format(i, batch_size_to_use, n_train)
        costs.append(theano_function(i, batch_size_to_use))
        i += batch_size_to_use

    return costs

def save_reformed_images(images, aa, img_shape=(19,25), tile_shape=(20,25),
                        original_image_fn=None, recreated_image_fn=None):
    assert len(images) == tile_shape[0] * tile_shape[1]

    originals = data_manager.tile_raster_images(
        X=images, img_shape=img_shape, tile_shape=tile_shape, tile_spacing=(1, 1))

    reformed = data_manager.tile_raster_images(
        aa.recreate(images), img_shape=img_shape, tile_shape=tile_shape,
        tile_spacing=(1, 1))

    if original_image_fn != None:
        img = Image.fromarray(originals)
        img.save(original_image_fn)

    if recreated_image_fn != None:
        img = Image.fromarray(reformed)
        img.save(recreated_image_fn)

if __name__ == "__main__":

    ########################
    # PREPARE DATA
    import data_manager
    train, labels, names = data_manager.load_train_set()

    eyes = np.array([l[0].ravel() for l in data_manager.build_eye_trainset(train, labels)])
    np.random.shuffle(eyes)
    n_train = len(eyes)

    inputs = theano.shared(eyes, borrow=True)

    ########################
    # BUILD AA

    n_in = np.shape(eyes[0])[0]

    HIDDEN_UNITS = 500
    LEARNING_RATE = 0.1

    with open("data/save/aa/single_layer/eyes_epoch_3000_291.5040.pkl","rb") as f:
        w, b_in, b_out = cPickle.load(f)

    aa = Autoencoder(inputs, n_in, HIDDEN_UNITS, LEARNING_RATE, W=w, b_in=b_in, b_out=b_out)

    ########################
    # TRAIN AA

    def run_epochs(aa, n_epochs, save=True):
        if 'n' not in dir(run_epochs):
            run_epochs.n = 0

        if 'costs' not in dir(run_epochs):
            run_epochs.costs = [(0, 999999)]

        start = time.time()
        print time
        for x in xrange(n_epochs):
            run_epochs.n += 1
            costs = epoch(1000, n_train, aa.train_step)
            print "=== epoch {} ===".format(run_epochs.n)
            print "costs: {}".format([line[()] for line in costs])
            print "avg: {}".format(np.mean(costs))
            run_epochs.costs.append((run_epochs.n, np.mean(costs)))

        elapsed = (time.time() - start)
        print "ELAPSED TIME: {}".format(elapsed)

        if save:
            vis = aa.visualize_hidden()
            img = Image.fromarray(vis)
            summary = "eyes_epoch_{}_{:0.4f}".format(run_epochs.n, run_epochs.costs[-1][1])
            img.save("imgs/aa/single_layer/{}.png".format(summary))
            aa.save("data/save/aa/single_layer/{}.pkl".format(summary))

        if run_epochs.costs[-2][1] < run_epochs.costs[-1][1]:
            aa = aa.copy(aa.learning_rate * 0.9)

        if run_epochs.n % 500 == 0:
            original_fn = "imgs/aa/single_layer/eyes_original.png"
            recreated_fn = "imgs/aa/single_layer/eyes_recreated_epoch_{}.png".format(run_epochs.n)
            save_reformed_images(eyes[:500], aa, original_image_fn=original_fn,
                                    recreated_image_fn=recreated_fn)


    # vis = aa.visualize_hidden()
    # img = Image.fromarray(vis)
    # img.save("imgs/aa/single_layer/eyes_aa_epoch_0.png")

    run_epochs.n = 3000
    run_epochs.costs = [(3000, 291.5040)]
    for i in xrange(150-30):
        run_epochs(aa, 100)

