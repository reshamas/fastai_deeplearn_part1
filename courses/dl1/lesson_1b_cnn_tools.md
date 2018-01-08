# Lesson 1b:  CNN and Technical Tools

(30-Oct-2017, live)  


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
* **State of the art** approach in many, if not most areas:  
   * image recognition
   * NLP
   * computer vision
   * speech recognition
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
- We take some set of pixels
- We multiply the pixels by some set of values (filter) and sum that up
- White areas have high numbers; black areas have low numbers

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

:key: **Note:**  Remember to <kbd>git pull </kbd> when in Crestle to ensure we're using the latest version of the repo.  

We're using:  [lesson1.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb) which is Image Classification with CNN 'Dogs vs Cats'

## fastai library
- `from fastai.imports imports import *` this imports a whole lot of other libraries
- has a bunch of deep learning libraries, it wraps around a lot of libraries
- the main library it is using is PyTorch
- all functions in fastai library are about 3-4 lines of code; they are short and designed to be understandable

See [notes on Juypter Notebook](/tools/jupyter_notebook.md) in tools directory.

## Setting up the data
* data needs to be in its folder by class

** add in notes for setting up data in directory

## Results
- With 20K images and 7 seconds, we were able to create a state-of-the-art classifier.  
- When people say you need Google-scale data or Google-scale infrastructure for deep learning, it's kind of meaningless.  
What they really mean is if you're working on exactly same problem, and using the exact same algorithm, at exactly the same time, if they train on more GPUs with more data for longer than you did, they will get a slightly better result.  
- But, if you want to get a good result on your problem, you can do it using a single GPU using a short time.  
- In this course, we will get a state-of-the-art results using a single GPU in a few seconds, minutes, maybe a couple of hours.

Note:  
- Crestle is a bit slower, the first time you run it.  It's also running on an older GPU.
- Jeremy is running it on a $600 GPU.  

### Model
- You can learn a lot about a dataset by analyzing what the model tells you.  
- Common complaint about deep learning is it's a black box.  That's definitely not true.  We can look at pictures and see what our model is looking at.  

This is the label for data:  
`data.val_y`  label for the validation set  
```bash
array([0,0,0,...1,1,1])
```
`data.classes`  0=cats and 1=dogs; can have as many categories as you like
```bash
['cats', 'dogs']
```
### Data Cleaning  
- we can look at data that are incorrectly classified
- Approach:  build model, find out what data needs to be cleaned.  
- we can look at "most correct dogs", "most correct cats"
- can look at "most incorrect"
- can look at "most uncertain predictions", sorted by how close to 0.5 the probability is

### Building a CNN
```python
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 3)
```
`resnet34` - architecture of convolutional neural network  
    - a bunch of different architectures
    - in practice, we need to pick one
    - may have heard that choosing architectures and hyperparameters takes a long time to learn.  This is largely UNTRUE.
We need to know 3 things:  
1.  architecture
2.  `learn.fit(0.01, 3)` --> **0.01** = learning rate
3.  `learn.fit(0.01, 3)` --> **3**  = number of epochs

- We've already figured out automatically all the other things you need (hyperparameters, etc) for you.  Turns out everything you need to choose are things that you can either automate or provide guidance on.  
- Computer vision - **resnet34** is the one to start with, and probably end with.  it is fast and accurate.  If you can't get a good result with resnet34, something else is probably wrong
- Number of Epochs - how many times do you want your algorithm to go through your images and read them?  **Start with 1**.  If that doesn't work, then run more.
- Learning Rate - this is the only one that is complex to pick.  With gradient descent, can pick out which direction to move downhill and take a small step in that direction.  Mathematically, we take the derivative of the function.  Learning rate is what do you multiply the derivative by.  
- Figuring out the learning rate has been the biggest challenge for practitioners.  

### Learning Rate
This researcher wrote a paper that shows a reliable way to set the learning rate every time:  
* [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) (WACV 2017) by Leslie Smith
* The learning rate determines how quickly or how slowly you want to update the weights (or parameters). Learning rate is one of the most difficult parameters to set, because it significantly affect model performance.
* We first create a new learner, since we want to know how to set the learning rate for a new (untrained) model.
```bash
learn = ConvLearner.pretrained(arch, data, precompute=True)
```
```bash
lrf=learn.lr_find()
learn.sched.plot_lr()
```
What it does, each iteration (dataset once), looks at a few images at a time (mini-batch), starts very small, and increases it over time.  Plot of loss (error) against the learning rate.  
- Pick the largest learning rate where it is clearly getting better.  

### Memory Issues
- Your GPU has a certain amount of memory on it.  
- If we try to pass in too many images at a time, it will run out of memory.
```python
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 3)
```
- number of images we pass in at a time is called batch size.
- Jeremy: 11 GB RAM
- AWS, Crestle:  12 GB RAM

If you're using a GPU with less space, you may get errors; you will need to use a smaller batch size. Can change `bs` from 64 to 32.  
In Jupyter Notebook:  hit Shift + Tab, after `(arch, sz), `
change `bs=32`  
Error is:  CUDA error
Change **batch size** and **restart kernel**.
Once the Jupyter Notebook has one GPU error, use it does not recover gracefully from it.  

## Homework
- use Jupyter Notebook, play with it
- can use Crestle (cheap if you don't use GPU switch), can play with Jupyter Notebook on Crestle
- make sure you can run all the steps in Crestle
- try to look at some pictures
- try different learning rates, how much slower is it? is it slower?
- can look at Lesson 2, which looks at satellite images, which has multiple labels (hazy primary, agriculture clear primary water)


## Next Week
#### Data Augmentation
- each time that the algorithm sees a cat picture, it will see a slightly different version of the picture.  (Example:  cats)
  - photo slightly to left, to right, flipped image
  - change image each time so algorithm gets different version of it, so it can recognize it in different versions
- to add data augmentation, just need to add these 2 parameters
  - `aug_tfms = transforms_side_on, max_zoom=1.1`
```python
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
```

 

