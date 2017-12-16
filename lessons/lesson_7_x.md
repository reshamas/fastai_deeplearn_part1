# Lesson 7  
live 11-Dec-2017

[Video: Lesson 7](https://www.youtube.com/watch?v=H3g26EVADgY&feature=youtu.be)

[Wiki: Lesson 7](http://forums.fast.ai/t/lesson-7-wiki-thread/8847/1)

Notebooks:  
* [lesson6-rnn.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson6-rnn.ipynb)
* [lesson7-cifar10.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson7-cifar10.ipynb)

---
## Other links
- WILD ML RNN Tutorial - http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
- Chris Olah on LSTM http://colah.github.io/posts/2015-08-Understanding-LSTMs/
- More from Olah and others - https://distill.pub/
- [BatchNorm paper](https://arxiv.org/pdf/1502.03167.pdf)
- [Laptop recommendation](https://youtu.be/EKzSiuqiHNg?t=1h1m51s); [Surface Book 2 15 inch](https://www.cnet.com/products/microsoft-surface-book-2/review/)


## Theme of Part 1
- classification and regression with deep learning
- identifying best practices
- here are 3 lines of code for image classification
- first 4 lessons were NLP, structured data, collaborative filtering
- last 3 lessons were above topics in more detail, more detailed code

## Theme of Part 2
- generative modeling
- creating a sentence, image captioning, neural translation
- creating an image, style transfer
- moving from best practices to speculative practices
- how to read a paper and implement from scratch
- does not assume a particular math background, but be prepared to dig through notation and convert to code

## RNN
- not so different
- they are like a fully connected network

## Batch Size
`bs=64` means data is split into 65 chunks of data.  
NOT batches of size 64!  

## Data Augmentation for NLP
- JH can't talk about that; doesn't know a good way
- JH will do further study on that

## CIFAR 10
- well-known dataset in academia:  https://www.cs.toronto.edu/~kriz/cifar.html
- small datasets are much more interesting than ImageNet
- often, we're looking at 32x32 pixels (example:  lung cancer image)
- often, it's more challenging, and more interesting
- we can run algorithms much more quickly, and it's still challenging
- you can get the data by:  `wget http://pjreddie.com/media/files/cifar.tgz` (provided in form we need)
- this is mean, SD per channel; try to replicate on your own
```python
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))
```  
- Kerem's notebook on how different optimizers work:  https://github.com/KeremTurgutlu/deeplearning/blob/master/Exploring%20Optimizers.ipynb
- to improve model, we'll next replace our fully connected model (with 1 hidden layer) with a CNN
- `nn.Conv2d(layers[i], layers[i + 1], kernel_size=3, stride=2)`
  - `layers[i]` number of features coming in
  - `layers[i + 1]` number of features coming out
  - `stride=2` is a "stride 2 convolution"
  - it has similar effect to `maxpooling`; reduces the size of the layers
- `self.pool = nn.AdaptiveMaxPool2d(1)` 
  - standard now for state-of-the-art algorithms
  - I'm not going to tell you how big an area to pool, I will tell you how big a resolution to create
  - starting with 28x28:  Do a 14x14 adaptive maxpool; same as 2x2 maxpool with a 14x14 output
  
 ## BatchNorm (Batch Normalization)
 - a couple of years old now
 - makes it easier to train deeper networks
 - 




