# Lesson 1b:  Deep Learning
(30-Oct-2017, live)  

### Video
* [Section 1](https://www.youtube.com/watch?v=sNMHZM2U7I8)  
* [Section 2](https://www.youtube.com/watch?v=ZDq5OXsLO3U)  

### Wiki
[Wiki: Lesson 1](http://forums.fast.ai/t/wiki-lesson-1/7011)  

### Notebooks Used  
[lesson1.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)  

--- 

## Deep Learning
* deep learning is a way of doing machine learning
* way of giving machine data (examples) and having it figure out the problem that is represented in those examples

## What We Are Looking For:  Something That Has 3 Properties
(3 Things that Give Us Modern Deep Learning)  
We are looking for a **mathematical function** that is *so flexible* that it can solve any given problem.  
1. Infinitely Flexible Functions
2. All-Purpose Parameter Fitting (way to train the parameters)
  * things can fit hundreds of millions of parameters
3. Fast and scalable

Example of limitation:  linear regression is limited by the fact it can only represent linear functions.  

Deep Learning has all 3 of above properties.  
* functional form:  neural network
* multiple layers allows more complex relationships
* parameters of neural network can be found using gradient descent

### Gradient Descent
* approach works well in practice; local minima are "equivalent" in practice
* different optimization techniques determine how quickly we can find the way down.

### Key discoveries thru Theoretical Side
* Very, very simple architectures of neural network and very, very simple methods of gradient descent work best in most situations.  
* We'll learn how every step works, using simple math.  

### Fast and Scalable:  Made Possible by GPUs
* GPU = Graphical Processing Unit
* GPUs are used in video games
* Huge industry of video games accidentally built for us what we need to do deep learning
* GPUs are useful and needed for deep learning
* GPUs are 10x faster than CPUs
* Best hardware for deep learning:  NVIDIA GTX-1080 Ti for ~ $600

## Art of Learning
* [A Mathematician's Lament](https://www.maa.org/external_archive/devlin/LockhartsLament.pdf) by Paul Lockhart (25 pages)
* [40 Years of Teaching Thinking: Revolution, Evolution, and What Next?](https://www.youtube.com/watch?v=-nmt1atA6ag) video, 2011 (1 hr 12 min)

## Projects Done
* [How HBO’s Silicon Valley built “Not Hotdog” with mobile TensorFlow, Keras & React Native](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3) by Tim Anglade

## Work
* will need to put in 10 hours a week (in addition to lecture time)
* spend time **RUNNING THE CODE** (rather than researching the theory)
* create blog posts 

## The Test of Whether You Can Understand
* Deep Learning is about solving problems
  * if you can't turned it into code, you can't solve the problem.  
* You can code / build something with it
* You can explain / teach it to someone else
  * Write a blog post
  * Help others who have questions
  
## Portfolio
* people are hired based on their portfolio (not USF DL certificate)
* GitHub projects, blog posts --> **can get hired based on portfolio**
* write down what you are learning in a form that other people can understand

## Goal
* main goal is not to help you move to a deep learning job
* continue doing what you're doing and bring deep learning to that
* examples:  medicine, journalism, dairy farming
* opportunities to change society
* focus:  help you be a great practitioner of deep learning
* opportunity - doing things differently
* come up with a project idea


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
Around much longer
Easier to use
Does NOT use multi-GPUs, but does everything else well

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
 
## CNN
The most important architecture for deep learning.
State of the art approach in most areas, image recognition, NLP, computer vision, speech recognition.
Best architecture for vast majority of applications
Basic structure is the convolution


http://setosa.io/ev/image-kernels/


The other architecture is Recurrent Neural Network.
Fully Connected NN

Each pixel is represented by a number from 0 to 255
White = 255
Black =small number, close to 0
We’re working with this matrix of numbers.

Let's walk through applying the following 3x3 sharpen kernel to the image of a face from above.
AKA:  filter

Convolution
We take some set of pixels
We multiply the pixels by some set of values (filter) and sum that up
White areas have high numbers; black areas have low numbers
Creates an EDGE DETECTOR
