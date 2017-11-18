# Lesson 1:  Deep Learning


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
