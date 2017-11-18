# Lesson 1b:  CNN
(30-Oct-2017, live)  

### Video
* [Section 1](https://www.youtube.com/watch?v=sNMHZM2U7I8)  
* [Section 2](https://www.youtube.com/watch?v=ZDq5OXsLO3U)  

### Wiki
[Wiki: Lesson 1](http://forums.fast.ai/t/wiki-lesson-1/7011)  

### Notebooks Used  
[lesson1.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)  

--- 

## CNN (Convolutional Neural Networks)
* the most important architecture for deep learning
* some researchers in the field think it's the only architecture we need
* the other main architectures we'll look at in this course are:
  * Recurrent Neural Network
  * Fully Connected Neural Networks
* **State of the art** approach in many, if not most areas:  image recognition, NLP, computer vision, speech recognition.
* this will be our focus:  best architecture for vast majority of applications
* basic structure of the CNN is the convolution

## [Convolution](https://en.wikipedia.org/wiki/Convolution)
In mathematics (and, in particular, functional analysis) convolution is a mathematical operation on two functions (f and g) to produce a third function, that is typically viewed as a modified version of one of the original functions, giving the integral of the pointwise multiplication of the two functions as a function of the amount that one of the original functions is translated.

This site explains it well visually:  
http://setosa.io/ev/image-kernels/

### How a Picture Becomes Numerical Data
* an image is made up of pixels
* each pixel is represented by a number from 0 to 255
  * White = 255
  * Black = 0 (small numbers, close to )
* we’re working with this matrix of numbers

#### Convolution
We take some set of pixels
We multiply the pixels by some set of values (filter) and sum that up
White areas have high numbers; black areas have low numbers

Let's walk through applying the following 3x3 sharpen kernel to the image of a face from above.  
AKA:  filter, pre-defined convolution filter
* any 3x3 matrix used to multiply a 3x3 area is called a kernel and the operation itself is called a convolution

Creates an EDGE DETECTOR

- it is a linear operation, only does multiplication and additions
- finds interesting features in image, like edges

What if we took convolutions and stacked them up, on top of each other? 
- We would have convolutions of convolutions, taking output of a convolution and input-ing it into another convolution?  
- That would actually *not be* interesting, because we're doing a linear function to another linear function.
- What *is* interesting, is if we put a **non-linear** function in between.

Example of non-linear function:  
* sigmoid

Michael Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com):    
http://neuralnetworksanddeeplearning.com/chap4.html

[Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)  
In the mathematical theory of artificial neural networks, the universal approximation theorem states[1] that a feed-forward network with a single hidden layer containing a finite number of neurons (i.e., a multilayer perceptron), can approximate continuous functions on compact subsets of Rn, under mild assumptions on the activation function. The theorem thus states that simple neural networks can represent a wide variety of interesting functions when given appropriate parameters; however, it does not touch upon the algorithmic learnability of those parameters.

Neural Network is:  
- linear function followed by some sort of non-linearity
- we can repeat that a few times

A common type of non-linearity:  ReLU (Rectified Linear Unit)  
`max(0, x)`  
- multiply numbers (kernel by a fixed frame)
- add the numbers up
- put thru non-linearity (ReLU):  set negative number to 0, leave positive number as is

## Filters
- can find edges, diagonals, corners
- vertical lines in the middle, left
- can find checkerboard patterns, edges of flowers, bits of text
- each layer finds multiplicatively more complex features
- dogs heads, unicycle wheels

What's next?  
- Nothing, that's it.
- Neural nets can approximate any function.
- There are so many functions.
- GPUs can do networks at scale; can do billions of operations a second.

Note:  most students spend lecture listening, then watch the lecture during the week and follow along with code.

## GPUs
- we need an NVIDIA GPU.  An NVIDIA GPUs support CUDA.  CUDA is a particular way of doing compution on GPU that is not limited to computer games, but general purpose computations.
- most laptops do not have NVIDIA GPUs.
- we need to access computers that access NVIDIA GPUs or computer specifically designed for deep learning.

## [Crestle](crestle.com)
Using notebook [lesson1.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb):  Image classification with Convolutional Neural Networks
- make sure instance is "GPU enabled"
- we're paying about $0.50 per hour

Jupyter Notebook - in [Kaggle data science survey](https://www.kaggle.com/surveys/2017) of 16K data scientists, Jupyter Notebook came up as 3rd most important self-reported tool for data science.  


## Winners of ImageNet
:key: [9 Deep Learning Papers - Need to Know](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)

### Architectures
- 2017 Nasnet, Google ? 
- 2016 Resnet, Microsoft
- 2015 GoogLeNet, Inception
- 2014 VGG Net, Oxford
   * Last of the really powerful simple architectures
   * VGG’s simpler approach is not much less accurate than others
   * For teaching purposes, it is pretty State of the art AND easy to understand
   * Excellent for the problems that differ (like satellite imagery vs simple photos)
- 2013 ZF Net, NYU (Matthew Zeiler and Rob Fergus)
- 2012 AlexNet, Stanford


### Layers

#### TensorFlow
Jeremy will not be talking about TensorFlow in this course.  It’s still very early, it’s still very new.
It does cool things, but not the kind of things uncool people need access to.

#### Theano
* has been around much longer
* easier to use
* eoes NOT use multi-GPUs, but does everything else well

#### Keras
If you build something in Keras, and you get to a point where it’s working great, want an extra 5%, it’s a simple configuration change to change backend to TensorFlow

Keras Configuration File
```bash
ubuntu@ip-10-0-0-13:~$ cd ~/.keras
ubuntu@ip-10-0-0-13:~/.keras$ ls
keras.json
ubuntu@ip-10-0-0-13:~/.keras$ 
ubuntu@ip-10-0-0-13:~/.keras$ cat keras.json                                        
{"image_dim_ordering": "th",
"epsilon": 1e-07,
"floatx": "float32",
"backend": "theano"
}
ubuntu@ip-10-0-0-13:~/.keras$ 
```
Change “th” to “tf”
Change “theano” to “tensoflow”



ubuntu@ip-10-0-0-13:~$ cat ~/.theanorc
[global]
device = gpu
floatX = float32
ubuntu@ip-10-0-0-13:~$ 

Note: device is either “cpu” or “gpu”

The t2 instance does not support GPUs.

We look at a few data files at a time, called them “batch” or “mini-batch”
GPUs need to runs lots of things at once
Single image is not enough to keep the GPU busy, so it gets slow

Why not do all data at once?
Only has a certain amount of memory:  about 2-12 GB of memory; our dataset will not fit into that
It’s not necessary to put all the data in to run it.
 

