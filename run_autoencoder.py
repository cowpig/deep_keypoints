import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import scale
import time
import matplotlib.pyplot as plt
from PIL import Image
import cPickle
import copy
import data_manager


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


########################
# PREPARE DATA

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

