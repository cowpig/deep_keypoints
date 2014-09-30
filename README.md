# deep_keypoints #

## Overview ##

This project is effectively a study in deep neural networks for computer vision. The dataset I'm using is from a kaggle's competition: http://www.kaggle.com/c/facial-keypoints-detection

It consists of faces, and the kaggle problem is to identify "keypoints" in the image: the center of the eyes, the tip of the nose, etc. Here's an example:

![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/face_with_keypoints.png)

Of course, the data isn't entirely clean, so there's a lot of cleanup code in `datamanager.py`. Note this image, which is missing some of the labels:

![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/face_missing_keypoints.png)

## Denoising Autoencoders ##

Denoising autoencoders (dAAs) do something very similar to what restricted boltzmann machines (RBMs) do, which is in turn similar to singular value decomposition. All three are unsupervised learning techniques that find ways to map some m-dimensional data into a new, n-dimensional space, such that descriptive features of the dataset are captured in the new set of dimensions.
Unlike SVD and other eigen-based methods, RBMs and dAAs can find curved manifolds in a space, and also map that space into *more* dimensions. See [this awesome blog article](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) for a nice visual intuition about this, and [this cool demo](http://cs.stanford.edu/people/karpathy/convnetjs//demo/classify2d.html) to see a neural network doing this with labeled data.
A dAA with fewer hidden nodes than inputs (that is, where n < m dimensions) can also be thought of as a lossy compression algorithm. Given some input data, an autoencoder finds the best way to map inputs to neurons such that the maximum amount of information about the input is preserved. 

To put it mathematically, an optimal autoencoder holds the following property true:

`f(f(input * w + b_in) * w + b_out) = input`

Where:
`f` is the [sigmoid function](http://mathworld.wolfram.com/SigmoidFunction.html)
`w` is the weight matrix that represents the connection strengths between the inputs and hidden nodes
`b_in` and `b_out` are the bias vectors

In practice, using denoising encoders to do unsupervised pretraining of a neural network tends to both improve training set costs, and reduce overfitting (and thus test set costs).

A denoising autoencoder works in much the same way as a typical single-layered neural network for classification, but with two significant differences:
* The original input is the expected output (as opposed to a one-hot vector)
* Noise is randomly added to the input vector (hence *denoising* autoencoder) before being passed into the network
The second part consists of setting random input units to zero, and forces the dAA to learn *dependencies between inputs*, and not just a way to map the inputs back onto themselves.

## Using dAAs for Eye Classification ##

One challenging aspect of the keypoint problem is that classifiers tend to have a very hard time distinguishing mouths from eyes in grainy black-and-white. To combat this, I thought that training an autoencoder to learn features specific to something like an eye would be interesting. 
It's pretty fascinating how a single-layer autoencoder filters images of eyes:

![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/eyes_original.png)

becomes:

![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/eyes_recreated_epoch_14500.png)

It's interesting how things like the rims of glasses are often filtered out of the image completely, or shades lightened to the point that it's possible to see through the lens. This seems to be on par with the way that human beings perceive visual data, and also serves as a powerful filter for extracting the features of an image that are most eye-like.

Here's a visual representation of the progression of the features through training:

![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/eyes_epoch_10.png)

*after 10 epochs, the filters are mostly just noise*

![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/eyes_epoch_100.png)

*after 100 epochs, slightly eye-like features begin to appear*

![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/eyes_epoch_1000.png)

*after 1000 epochs, there remain many noisy features, but the general contours of eyes are being modelled, as well as irises*

![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/eyes_epoch_14500.png)

*after 14500 epochs, many features focus in some way on the contour of the eye, the iris or white of the inside, or skin folds. Almost all of the features are visibly identifiable as eye features*

And indeed, the error rate for the eye classifier drops significantly when the weights of the neural network are pretrained with an autoencoder:
![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/compare_networks.png)
A error rate drop from about 7.5% to about 4.9% is a massive improvement.

## Convolutional Networks ##

Unfortunately, I've never found a good resource explaining how convolutional networks work, so [until I finish writing my own](https://github.com/cowpig/SuperVision-Explained), I'm just going to skip the explanation phase.

Besides, it turns out that convolutional networks are really computationally expensive. For a 96x96 image, 50 convolutional features, and 5x5 subsampling, a single layer produces 96*96*50*5*5 = 11.5 million outputs. Connect this single layer to a 96x96 matrix of keypoints and we have on the order of hundreds of billions of computations per image. Obviously this becomes pretty computationally expensive: on an x-large 32-core amazon server, this setup ran for over 48 hours without finishing training. The output at that point certainly looked promising, however.
Enter `theano`'s gpgpu capability...

## TODO: ##

* get a the convolutional network trained on gpu
* do cascading to deal with memory issues?
