# Lesson 1:  Intro

(30-Oct-2017, live)  

### Video
* [Section 1](https://www.youtube.com/watch?v=sNMHZM2U7I8)  
* [Section 2](https://www.youtube.com/watch?v=ZDq5OXsLO3U)  

### Wiki
[Wiki: Lesson 1](http://forums.fast.ai/t/wiki-lesson-1/7011)  

### Notebooks Used  
[lesson1.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)  

--- 

## USF
In-person Info:  [Deep Learning Certificate Part I](https://www.usfca.edu/data-institute/certificates/deep-learning-part-one)  

### Staff
* Intro by [David Uminsky](https://www.usfca.edu/faculty/david-uminsky), Director of Data Institute of USF 
* [Yannet Interian](https://www.usfca.edu/faculty/yannet-interian), Assistant Professor USF
* [Rachel Thomas](https://www.usfca.edu/data-institute/about-us/researchers), Researcher in Residence
* [Jeremy Howard](https://www.usfca.edu/data-institute/about-us/researchers), Distinguished Scholar in Deep Learning

### Classroom
* 200 students in room at USF
* 100 Masters' students upstairs at USF
* 400 International Fellows, via livestream

Being recorded and will become a fastai MOOC.  

#### Teams
* teams of 6 people
* get help with stuff

## Python
* using Python 3.6

## Platforms
* [Crestle](https://www.crestle.com) built by [Anurag Goel](https://www.linkedin.com/in/anuragoel/)  
* [Paperspace](https://www.paperspace.com)
* [AWS](https://aws.amazon.com/console/)

## Deep Learning
* Deep learning is a particular way of doing machine learning
* [Arthur Samuels](https://en.wikipedia.org/wiki/Arthur_Samuel)
  * he invented machine learning
  * rather than programming, step-by-step, give the computer *examples*
    * **let the computer figure out the problem by giving it examples**
  * let computer play checkers against itself thousands of times; it figured out which parameters worked the best
  * Samuel **Checkers-playing** Program appears to be the world's first self-learning program, and as such a very early demonstration of the fundamental concept of artificial intelligence (AI); 1962
  * he worked at Bell Labs and IBM, then Stanford Univ
  
### Machine Learning
#### Example:  ML Algorithm in Predicting Breast Cancer Survival Based on Pathology Slides
* start with pictures of breast cancer slides
* work with computer scientists, pathologists worked together to determine features that would predict who would survive or not, based on slides
* process of building model can take some time (many years); can pass data into ML algorithm, such as logistic regression; regression can determine which sets of features separate out the 2 classes
* this can work well, but requires a lot of experts and requires the feature data
* this ML algorithm was more accurate at predicting breast cancer survival than human pathologists 

#### Examples of ML Uses, Thanks to Deep Learning
* gmail, generates automatic responses to emails.
* skype, translate to different languages, in real time
* At Google, every single part of the company uses deep learning
* [DeepMind AI Reduces Google Data Centre Cooling Bill by 40%](https://deepmind.com/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-40/)
* [Baidu’s Deep-Learning System is better at English and Mandarin Speech Recognition than most people](https://www.nextbigfuture.com/2015/12/baidus-deep-learning-system-is-better.html)

#### Future Work
How do we get computers and humans to work better together?   

#### Goal of This Course
* that people from all different backgrounds will use deep learning to solve problems

## [ImageNet](http://www.image-net.org)
First step is to use a pre-trained model.  

### Pre-trained Model:  
* Someone has already come along, downloaded millions of images off of the internet
* Built a deep learning model that has learned to recognize the contents of those images
* Nearly always, with these pre-trained models, they use ImageNet dataset
* ImageNet has most respected annual computer vision competition (winners are Google, Microsoft)
  * 32,000+ categories
* Folks that create these pre-trained networks basically download large subset of images from ImageNet
* Shortcomings of ImageNet Dataset
  * ImageNet is carefully curated so that photo has one main item in it
  * For us, this is a suitable dataset
* Each year, the winner make source code / weights available

## 3 Things that Give Us Modern Deep Learning
1. Infinitely Flexible Functions
2. Way to train the parameters
3. Fast and scalable

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
 
CNN
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
