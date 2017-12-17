## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[rep-images]: ./images/images.png "rep"
[graph]: ./images/graph.png "graph"
[tensorboard]: ./images/tensorboard.png "tensorboard"

### Overview

This is the second project in Udacity Self-Driving Car NanoDegree Term1. I implemented Traffic Sign Recognizer using Tensorflow.

I implemented a classifier using batchnorm provided by tensorflow slim in this project and analyzed the results using tensorboard.

### Prerequisites

* python 3.5
* tensorflow 0.12.1
* opencv 3.1.0
* etc.

A list of all the packages needed to run this project can be found in [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit). Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Description

I implemented the traffic image classifier using the Tensorflow framework.


#### 1. Network Architecture

Inspired by VGGnet, I designed the following model:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64    |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 					|
| Fully connected		| outputs 256        							|
| Fully connected		| outputs 43        							|
| Softmax				|         										|


Batch normalization was used for fully connected layers and convolutional layers except for final fully connected layer.

Here is the operation graph generated by tensorboard

![alt text][graph]

#### 2. Result

My final model results were:
* training set accuracy of 0.9998
* validation set accuracy of 0.9848
* test set accuracy of 0.9758

![alt text][tensorboard]

The source code for the experiment can be found in [Traffic_Sign_Classifier.ipynb](https://github.com/penny4860/CarND-P2-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)


### Dataset

Download dataset from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)

* The size of training set is (34799, 32, 32, 3)
* The size of the validation set is (4410, 32, 32, 3)
* The size of test set is (12630, 32, 32, 3)
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

The following figure shows a sample for each class. One sample was selected for each class and histogram equlization for each channel was performed and plotted.

![alt text][rep-images]
