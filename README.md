# deep_keypoints #
==============

## What ##

This project is effectively a study in deep neural networks for computer vision. The dataset I'm using is from a kaggle's competition: http://www.kaggle.com/c/facial-keypoints-detection

It consists of faces, and the kaggle problem is to identify "keypoints" in the image: the center of the eyes, the tip of the nose, etc. Here's an example:
![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/face_with_keypoints.png)
Of course, the data isn't entirely clean, so there's a lot of cleanup code in `datamanager.py`. Note this image, which is missing some of the labels:
![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/face_missing_keypoints.png)

## Denoising Autoencoder ##

One challenging aspect of the keypoint problem is that classifiers tend to have a very hard time distinguishing mouths from eyes in grainy black-and-white. To combat this, I thought that training an autoencoder to learn features specific to something like an eye would be interesting. 
It's pretty fascinating how a single-layer autoencoder filters images of eyes:
![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/eyes_original.png)
becomes:
![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/eyes_recreated_epoch_14500.png)
Interesting how things like the rims of glasses are often filtered out of the image completely, or shades lightened to the point that it's possible to see through the lens. This seems to be on par with the way that human beings perceive visual data, and also serves as a powerful filter for extracting the features of an image that are most eye-like.
And indeed, the error rate for the eye classifier drops significantly when the weights of the neural network are pretrained with an autoencoder:
![alt tag](https://raw.github.com/cowpig/deep_keypoints/master/imgs/compare_networks.png)
A error rate drop from about 7.5% to about 4.9% is a massive improvement.

## Convolutional Networks ##

It turns out that convolutional networks are really computationally expensive. For a 96x96 image, 50 convolutional features, and 5x5 subsampling, a single layer produces 96*96*50*5*5 = 11.5 million outputs. Connect this single layer to a 96x96 matrix of keypoints and we have on the order of hundreds of billions of computations per image. Obviously this becomes pretty computationally expensive: on an x-large 32-core amazon server, this setup ran for over 48 hours without finishing training. The output at that point certainly looked promising, however.
Enter `theano`'s gpu capability...

## TODO: ##

- train the convolutional network with gpu
- do cascading to deal with memory issues