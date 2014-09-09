deep_keypoints
==============

Trying out deep learning methods on the facial keypoint recognition problem


TODO:

- add trainer class (refactoring necessary)
- fix everything to work with gpu
	- maybe look a bit more into optimizing code for gpu training
- add & test efficacy of momentum, weight decay, learning rate decay 
	-(Kryschevsky paper claims weight decay improved training error. This is surprising-- can I replicate this result?)
- try conv autoencoder
- do cascading to deal with memory issues
- finally submit something to kaggle & compare scores!