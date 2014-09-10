import cPickle
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.misc import imresize


BAD_IMAGES = [1747, 1907, 2199]
EYE_WIDTH = 24
EYE_HEIGHT = 18
MOUTH_WIDTH = 44
MOUTH_HEIGHT = 22
TOO_CLOSE_VALUE = 9

KEYPOINT_DICT = {
	"left_eye_center"      	  : (1,0),
	"right_eye_center"     	  : (3,2),
	"left_eye_inner"       	  : (5,4),
	"left_eye_outer"       	  : (7,6),
	"right_eye_inner"      	  : (9,8),
	"right_eye_outer"      	  : (11,10),
	"left_eyebrow_inner"   	  : (13,12),
	"left_eyebrow_outer"   	  : (15,14),
	"right_eyebrow_inner"  	  : (17,16),
	"right_eyebrow_outer"  	  : (19,18),
	"nose_tip"             	  : (21,20),
	"left_mouth_corner"    	  : (23,22),
	"right_mouth_corner"   	  : (25,24),
	"center_mouth_top_lip" 	  : (27,26),
	"center_mouth_bottom_lip" : (29,28)
}
COMMON_KEYPOINTS = ['left_eye_center', 'right_eye_center', 
							'nose_tip', 'center_mouth_bottom_lip']

# this will take an image of wxh dimensions and create cut**2 (w-cut)x(h-cut)
#	new images, and match the labels to these subimages. This way a dataset can be
#	massively increased in size
# assumes x, y are both lists of matrices that are the same dimensions
def create_cuts(x, y, cut):
	new_x = []
	new_y = []
	w, h = x[0].shape
	for img, label in zip(x, y):
		for i in xrange(cut):
			for j in xrange(cut):
				new_x.append(img[i:w-i, j:j-i])
				new_y.append(label[i:w-i, j:j-i])
	return (new_x, new_y)


def str_to_float(string):
	if string == '':
		return None
	return float(string)

# loads the training set and processes it. Outputs two 2d arrays: 
#   the raw image data, and the facial feature coordinates. the third
#   array it outputs is the list of the names of the facial features
#   (e.g. 'left_eye_outer_corner_x') I generally call the values 'labels'
#   and that refers to the number, not the string
def load_train_set(filename="data/training.csv", required_keypoints=None):
	train_set = []
	labels = []

	with open(filename, 'rb') as f:
		r = csv.reader(f)
		label_names = r.next()[:-1]

		for i, line in enumerate(r):
			if i not in BAD_IMAGES:
				label = [str_to_float(s) for s in line[:-1]]
				if required_keypoints != None:
					bad = False
					for kp in required_keypoints:
						for index in KEYPOINT_DICT[kp]:
							if label[index] == None:
								bad = True
					if bad:
						continue

				labels.append(label)
				unrolled = np.array([float(s) for s in line[-1].split(' ')])
				train_set.append(to_matrix(unrolled) / 255.)

	return (train_set, labels, label_names)

def build_keypoint_map(labels, active_keypoints=None):
	if active_keypoints == None:
		active_keypoints = ['left_eye_center', 'right_eye_center', 
							'nose_tip', 'center_mouth_bottom_lip']
	if active_keypoints == "all":
		active_keypoints = KEYPOINT_DICT.keys()

	output = []
	for i, label in enumerate(labels):
		next_map = np.zeros((96,96))
		for kp in active_keypoints:
			try:
				i_col, i_row = KEYPOINT_DICT[kp]
				row = np.round(label[i_row])
				col = np.round(label[i_col])
				row = min(row, 95)
				col = min(col, 95)
				next_map[row, col] = 1
			except:
				import pdb; pdb.set_trace()
		output.append(next_map)
	return output

def quarter_size_keypoint_map(keypoint_map):
	output = []
	for kpm in keypoint_map:
		new_kpm = zeros((48,48))
		for i in xrange(48):
			for j in xrange(48):
				new_kpm[i,j] = max(kpm[i*2:(i+1)*2, j*2:(j+1)*2])
		output.append(new_kpm)
	return output

def add_horizontal_flips(train, keypoint_map):
	train2 = []
	kpm2 = []
	for i in xrange(len(train)):
		train2.append(train[i])
		train2.append(flip_horizontal(train[i]))
		kpm2.append(keypoint_map[i])
		kpm2.append(flip_horizontal(keypoint_map[i]))
	return (train2, kpm2)

def full_trainset():
	train, labels, label_names = load_train_set(required_keypoints=COMMON_KEYPOINTS)
	kpm = build_keypoint_map(labels, COMMON_KEYPOINTS)
	return add_horizontal_flips(train, kpm)

def shuffle_all(*args):
	idx = range(len(args[0]))
	random.shuffle(idx)
	for arg in args:
		assert(len(arg) == len(args[0]))
	return [[arg[i] for i in idx] for arg in args]


# takes a line containing raw image data, and reshapes it into a 96 row,
#   96-column matrix
def to_matrix(line):
	assert(len(line) == 96 * 96)
	return np.reshape(line, (96, 96))

# takes an image and displays it
def display_image(img, label=None):
	if len(img) == 96*96:
		plt.imshow(to_matrix(img))
	else:
		plt.imshow(img)
	plt.gray()
	if label != None:
		x = []
		y = []
		pairs = zip(label[::2], label[1::2])
		for pair in pairs:
			if None not in pair:
				x.append(pair[0])
				y.append(pair[1])
		plt.scatter(x,y)
	plt.show()

# takes an image and displays it
def save_image(img, fn):
	if len(img) == 96*96:
		plt.imshow(to_matrix(img))
	else:
		plt.imshow(img)
	plt.gray()
	plt.savefig(fn)

# calculates the some stats about a set (or list) of label(name)s
def stats(labels, label_names, labels_to_check):
	# can take either the names of features or their indices
	if type(labels_to_check[0]) != int:
		label_indices = [label_names.index(name) for name in labels_to_check]
	else:
		label_indices = labels_to_check

	good = []
	bad = 0
	n = len(label_indices)

	# count images that are missing the labels
	for line in labels:
		good_line = True
		for i in label_indices:
			if line[i] == None:
				bad += 1
				good_line = False
				break
		if good_line:
			good.append(line)

	# get some statistics on a feature
	counts = {}
	for index in label_indices:
		counts[index] = []

	for line in good:
		for index in label_indices:
			counts[index].append(line[index])

	stats = {}
	for index in label_indices:
		name = label_names[index]
		stats[name] = {}
		stats[name]["avg"] = sum(counts[index]) / float(len(counts[index]))
		stats[name]["min"] = min(counts[index])
		stats[name]["max"] = max(counts[index])
		

	return {
		'num_missing' : bad,
		'individual_stats' : stats
	}


def resize(img, size):
	return ndimage.interpolation.zoom(img, size)

def get_subimage(img, top_left, bot_right):
	if len(img) == 96*96:
		img = to_matrix(img)
	top, left = top_left
	bot, right = bot_right
	return img[top:bot+1, left:right+1]

def euclidean_distance(a, b):
	if type(a) == tuple or type(a) == list:
		a = np.array(a)
	if type(b) == tuple or type(b) == list:
		b = np.array(b)
	return np.linalg.norm(a - b)

def label_distance(label, indices_a, indices_b):
	point_a = [label[indices_a[0]], label[indices_a[1]]]
	if point_a[0] == '' or point_a[1] == '' or point_a[0] == None or point_a[1] == None:
		return None
	point_b = [label[indices_b[0]], label[indices_b[1]]]
	if point_b[0] == '' or point_b[1] == '' or point_b[0] == None or point_b[1] == None:
		return None

	try:
		point_a = np.array([float(x) for x in point_a])
		point_b = np.array([float(x) for x in point_b])
	except:
		import pdb;pdb.set_trace()

	return euclidean_distance(point_a, point_b)

def flip_horizontal(matrix):
	if type(matrix) == list:
		return [row[::-1] for row in matrix]

	return matrix[...,::-1]

def build_eye_trainset(train_set, labels, add_negatives=False):
	# to_shuffle = zip(train_set, labels)
	# np.random.shuffle(to_shuffle)
	# train_set, labels = zip(*to_shuffle)

	eyes = []

	left_eye_inner = KEYPOINT_DICT['left_eye_inner']
	left_eye_outer = KEYPOINT_DICT['left_eye_outer']
	right_eye_inner = KEYPOINT_DICT['right_eye_inner']
	right_eye_outer = KEYPOINT_DICT['right_eye_outer']

	for i, label in enumerate(labels):
		dist_h_left_eye = label_distance(label, left_eye_inner, left_eye_outer)
		dist_h_right_eye = label_distance(label, right_eye_inner, right_eye_outer)

		# add each eye image with a positive label
		if dist_h_left_eye != 0 and dist_h_left_eye != None:
			left = label[4]
			right = label[6]
			middle = np.average([label[5], label[7]])

			padding = (EYE_WIDTH - (right - left))

			left = left - padding/2.
			right = right + padding/2.
			top = middle - EYE_HEIGHT/2.
			bot = middle + EYE_HEIGHT/2.

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			subimg = get_subimage(train_set[i], (top, left), (bot, right))

			if np.shape(subimg) != (19,25):
				print "{}, left".format(i)
				import pdb; pdb.set_trace()

			tl_l = (top, left)
			br_l = (bot, right)

			eyes.append((subimg, 1, i))
			eyes.append((flip_horizontal(subimg), 1, i))

		if dist_h_right_eye != 0 and dist_h_right_eye != None:
			left = label[10]
			right = label[8]
			middle = np.average([label[9], label[11]])

			padding = (EYE_WIDTH - (right - left))

			left = left - padding/2.
			right = right + padding/2.
			top = middle - EYE_HEIGHT/2.
			bot = middle + EYE_HEIGHT/2.

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			# deals with two specific outlier cases
			if (i == 1964) or (i == 2189):
				left += 1
				right += 1

			subimg = get_subimage(train_set[i], (top, left), (bot, right))

			if np.shape(subimg) != (19,25):
				print "{}, right".format(i)
				import pdb; pdb.set_trace()

			tl_r = (top, left)
			br_r = (bot, right)
			eyes.append((flip_horizontal(subimg), 1, i))
			eyes.append((subimg, 1, i))

		if add_negatives:
			def random(x):
				return int(np.random.random() * x)

			def too_close(new, *others):
				for other in others:
					if euclidean_distance(new, other) < TOO_CLOSE_VALUE:
						return True
				return False

			for _ in xrange(1):
				tl = (random(96 - EYE_HEIGHT), random(96 - EYE_WIDTH))
				br = (tl[0] + EYE_HEIGHT, tl[1] + EYE_WIDTH)

				while too_close(tl, tl_l, tl_r) or too_close(br, br_l, br_r):
					tl = (random(96 - EYE_HEIGHT), random(96 - EYE_WIDTH))
					br = (tl[0] + EYE_HEIGHT, tl[1] + EYE_WIDTH)

				eyes.append((get_subimage(train_set[i], tl, br), 0))

	return eyes

def build_mouth_trainset(train_set, labels, add_negatives=False):
	# to_shuffle = zip(train_set, labels)
	# np.random.shuffle(to_shuffle)
	# train_set, labels = zip(*to_shuffle)

	mouth_left_corner = (22, 23)
	mouth_right_corner = (24, 25)
	mouth_center_top_lip = (26, 27)
	mouth_center_bottom_lip = (28, 29)

	mouths = []
	distances = []

	for i, label in enumerate(labels):
		# mouth is just too close to the border in this image
		if i==1964:
			continue
		dist_h = label_distance(label, mouth_left_corner, mouth_right_corner)
		dist_v = label_distance(label, mouth_center_top_lip, mouth_center_bottom_lip)

		# add each eye image with a positive label
		if dist_h != 0 and dist_h != None and dist_v != 0 and dist_v != None and dist_h < 40:
			left = label[24]
			right = label[22]
			top = label[27]
			bot = label[29]

			padding_h = (MOUTH_WIDTH - (right - left))
			padding_v = (MOUTH_HEIGHT - (bot - top))

			# import pdb; pdb.set_trace()

			left = left - padding_h/2.
			right = right + padding_h/2.
			top = top - padding_v/2.
			bot = bot + padding_v/2.

			left = int(np.round(left))
			right = int(np.round(right))
			top = int(np.round(top))
			bot = int(np.round(bot))

			while (bot > 95):
				top -= 1
				bot -= 1

			subimg = get_subimage(train_set[i], (top, left), (bot, right))
			tl_m = (top, left)
			br_m = (bot, right)

			if subimg.shape != (23, 45):
				import pdb; pdb.set_trace()

			mouths.append((subimg, 1, i))
			mouths.append((flip_horizontal(subimg), 1, i))

			if add_negatives:
				def random(x):
					return int(np.random.random() * x)

				def too_close(new, *others):
					for other in others:
						if euclidean_distance(new, other) < TOO_CLOSE_VALUE:
							return True
					return False

				tl = (random(96 - MOUTH_HEIGHT), random(96 - MOUTH_WIDTH))
				br = (tl[0] + MOUTH_HEIGHT, tl[1] + MOUTH_WIDTH)

				while too_close(tl, tl_m, (0,0)) or too_close(br, br_m, (0,0)):
					tl = (random(96 - MOUTH_HEIGHT), random(96 - MOUTH_WIDTH))
					br = (tl[0] + MOUTH_HEIGHT, tl[1] + MOUTH_WIDTH)

				mouths.append((get_subimage(train_set[i], tl, br), 0))

	return mouths

def eyes_with_mouths_set():
	trainset, labels, names = load_train_set()
	eyes_and_randoms = build_eye_trainset(trainset, labels, True)
	mouths = build_mouth_trainset(trainset, labels)
	for mouth in mouths:
		eyes_and_randoms.append((imresize(mouth[0], (19,25)), mouth[1], 0))

	return eyes_and_randoms


def cross_sample(mouths, eyes):
	# the eyes sample is several times larger than the mouth
	num_mouths = len(mouths) * 0.5
	num_eyes = len(eyes) * 0.5

	eyeshape = np.shape(eyes[0][0])
	mouthshape = np.shape(mouths[0][0])

	for i in xrange(1, int(num_eyes*0.1), 2):
		try:
			eyes[i] = (imresize(mouths[i+1][0], eyeshape), 0)
		except Exception as e:
			print e.message
			import pdb; pdb.set_trace()

	for i in xrange(1, int(num_mouths*0.1), 2):
		mouths[i] = (imresize(eyes[i+1][0], mouthshape), 0)

	np.random.shuffle(mouths)
	np.random.shuffle(eyes)

def generate_patches(dataset, size=(EYE_WIDTH, EYE_HEIGHT), n_patches=10000):
	# assumes dataset is 96x96 images
	width = 96
	height = 96
	w, h = size
	out = []
	for i in xrange(n_patches):
		n = int(random.random() * len(dataset))
		x =  int(random.random() * (width - w))
		y =  int(random.random() * (height - h))
		d = dataset[n][x:x+w, y:y+h].ravel()
		out.append(d)

	return np.array(out)

def rescale(ndar, eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=False,
                       output_pixel_vals=True):

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]


    H, W = img_shape
    Hs, Ws = tile_spacing

    dt = X.dtype
    if output_pixel_vals:
        dt = 'uint8'
    out_array = np.zeros(out_shape, dtype=dt)

    for tile_row in xrange(tile_shape[0]):
        for tile_col in xrange(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                this_x = X[tile_row * tile_shape[1] + tile_col]
                if scale_rows_to_unit_interval:
                    this_img = rescale(
                        this_x.reshape(img_shape))
                else:
                    this_img = this_x.reshape(img_shape)
                c = 1
                if output_pixel_vals:
                    c = 255
                out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c

    return out_array

def visualize_hidden(weights, img_shape=(19,25), vis_shape=(25,20)):
    img_h, img_w = img_shape
    n_imgs_vert, n_imgs_hor = vis_shape

    assert n_imgs_hor * n_imgs_vert == self.n_hidden

    if type(W) == theano.tensor:
        weights = weights.get_value(borrow=True).T

    return data_manager.tile_raster_images(
        X=weights, img_shape=img_shape, tile_shape=vis_shape,
        tile_spacing=(1, 1))