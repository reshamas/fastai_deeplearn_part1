# Lesson 3:  xxx
(13-Nov-2017, live)

[Video: Lesson 3](https://www.youtube.com/watch?v=9C06ZPF8Uuc&feature=youtu.be) 

[Wiki: Lesson 3](http://forums.fast.ai/t/wiki-lesson-3/7809)  

## Notebooks Used
* dogs vs cats [lesson1-rxt50](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1-rxt50.ipynb)
* planet:  [lesson2-image_models.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson2-image_models.ipynb)  
* dogs vs cats using keras library:  [`keras_lesson1.ipynb`](https://github.com/fastai/fastai/blob/master/courses/dl1/keras_lesson1.ipynb)  
---

## Technical Resources
Repository:  https://github.com/reshamas/fastai_deeplearn_part1

#### AWS AMI Setup
Step-by-step instructions are here: [aws_ami_gpu_setup.md](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/aws_ami_gpu_setup.md)

#### Using tmux on AWS
Step-by-step instructions here [tmux.md](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/tmux.md)

#### Blogs
List of blogs can be found here:  [resources.md](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/resources.md)

---
## Where We Go From Here
1. CNN Image Intro
2. Structured Neural Net Intro
   - logistics, finance
3. Language RNN Intro
4. Collaborative Filtering Intro
   - recommendation systems
5. Collaborative Filtering In-depth
6. Structured Neural Net in Depth
7. CNN Image in Depth
8. Language RNN in Depth

---
## Learning How to Download Data
- Kaggle
- other places on internet

#### Kaggle CLI (command line interface)
Instructions on using `kaggle-cli`:  [kaggle_cli.md](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/kaggle_cli.md)  
Kaggle Competition:  [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

#### Chrome CurlWget
Instructions on using Chrome CurlWget:  [chrome_curlwget](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/chrome_curlwget.md)

---
## Setting up data directory
- fastai library assumes data is in a subdirectory where notebook is:  `PATH = "data/dogscats"`
- you might want to put the data somewhere else, such as your home directory
- you can put data anywhere you like, and use symlinks

#### Symbolic Links
Here’s an example. Let’s say you wanted to create a symbolic link in your Desktop folder that points to your Downloads folder. You’d run the following command:

<kbd> ln -s /Users/name/Downloads /Users/name/Desktop </kbd>

[How to Create and Use Symbolic Links (aka Symlinks) on a Mac](https://www.howtogeek.com/297721/how-to-create-and-use-symbolic-links-aka-symlinks-on-a-mac/)

In Linux, can do <kbd> ls -l dir_name </kbd> to see what the symlinks point to

---
## Quick Dogs vs Cats
* dogs vs cats [lesson1-rxt50](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1-rxt50.ipynb)

This code imports all the fastai libraries:  
```python
from fastai.conv_learner import *
PATH = "data/dogscats/"
sz=224; bs=48
```

#### Learning Rate Finder
This step assumes you have run the learning rate finder:  
```python
%time learn.fit(1e-2, 3, cycle_len=1)
```

#### `precompute=True` / data augmentation
```python
sz=299
arch=resnet50
bs=28
```
```python
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=4)
#learn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5)
learn = ConvLearner.pretrained(arch, data)
%time learn.fit(1e-2, 3, cycle_len=1)
```
- something that makes model step faster
- if you're confused about this, exclude this step. it's a shortcut which caches intermediate steps which don't have to be calculated each time.
- when we are using precomputed activations, data augmentation does not work.  Because `precompute=True` is using the cached, non-augmented activations.
- if you ask for data augmentation and have `precompute=True`, it doesn't actually do any data augmentation, because it is using the cached non-augmented activations

#### Unfreezing    
`learn.unfreeze()`
- now unfreeze so we can train the whole thing

`learn.bn_freeze`  Batch Norm Unfreeze
- If you're using a bigger, deeper model like resnet50 or resnext101, on a dataset that is very similar to ImageNet; This line should be added when you unfreeze.  This causes the batch normalization moving averages to not be updated. (more in second half of course)
- not supported by another library, but turns out to be important
- if you are using an architecuture with larger than 34 suffix (resnet50, resnext101), and you're training dataset with photos similar to ImageNet (normal photos, normal size, object in middle of photo and takes up most of frame), then you should add `bn_freeze`.  If in doubt, try with and without it.  

#### Re-train Network After Unfreezing
After unfreezing, train another epoch:   
```python
learn.unfreeze()
learn.bn_freeze
%time learn.fit([1e-5, 1e-4, 1e-3], 1, cycle_len=1)
```

#### Test Time Augmentation
`TTA` Test Time Augmentation - use to ensure we get the best predictions we can.  
```python
%time log_preds, y = learn.TTA()
metrics.log_loss(y, np.exp(log_preds)), accuracy(log_preds, y)
```

## Summary of Steps
Assuming:  
- you have run the learning rate finder
- data is set up in the director structure
- 

#### When you a try a new dataset, these are the minimum steps to take.  

```python
from fastai.conv_learner import *
PATH = "data/dogscats/"
```
```python
arch = resnet50
sz = 224
bs = 64
```
```python
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)
learn = ConvLearner.pretrained(arch, data)
%time learn.fit(1e-2, 3, cycle_len=1)
```
```python
learn.unfreeze()
learn.bn_freeze(True)
%time learn.fit([1e-5, 1e-4, 1e-3], 1, cycle_len=1)
```
```python
%time log_preds, y = learn.TTA()
metrics.log_loss(y, np.exp(log_preds)), accuracy(log_preds, y)
```

## Comparing fastai to Keras with TensorFlow
- fastai library sits on top of PyTorch.  
- keras sits on top of a variety of backends:  TensorFlow, MxNet, Microsoft CNTK

dogs vs cats using keras library:  [`keras_lesson1.ipynb`](https://github.com/fastai/fastai/blob/master/courses/dl1/keras_lesson1.ipynb)  
- Jeremy has attempted to replicate parts of lesson 1 in keras

#### Keras code
- import a bunch of libraries
- keras also has standard directory structure for image data:  training, valid
- specify what batch size to use
- we need to use much, much more code; each part of code has many things we need to set
- data generation:  specify type of data augmentation, normalization (fastai says whatever resnet50 requires)
- no standard set of best data augmentation parameters
- class mode, specify binary or multiclass
- need to do data generator for validation set
- with validation set, it's vital that data are not shuffled, but with training, they should be shuffled
- keras does not have resnet34
- keras, need to manually construct the layers
- in keras, need to compile model (not needed in fastai or pytorch, we know what is a good loss to use)
- no concept of automatically freezing layers
- need to tell it how many batches per epoch
- much more code, so more opportunities for error
- no concept of layer groups, differential learning rates or partial unfreezing
- manually had to define which layers to fine tune, then recompile model
- LOT MORE CODE & PERFORMANCE IS DIFFERENT

#### Comparison of Results:  
keras:  97% after 4 epochs, 8 minutes  
fastai:  99.5% on validation, 4-5 minutes

If you want to run this notebook, install as follows, since it is not part of the fastai environment:  
```pip install tensorflow-gpu keras```  

Deploying on Mobile:  
- Pytorch on mobile situation is early, may want to use TensorFlow
- will take work to get state-of-the-art results

what fastai library contains that is not in keras:  
- stochastic gradient with restart
- differential learning rates
- batch norm freezing

Keras and TensorFlow are not difficult to handle; would take a couple of days to learn.  
Google is interested in getting fastai library ported to TensorFlow.  

---
## Dog Breeds
- assignment from last week, try to do everything you've seen already, but do it on the dog breeds dataset
- last few minutes of last week:  code presented on how to look at data, classes, how big the images are, etc.  

### How to Submit to Kaggle
Every Kaggle competition has an "Evaluation" section with instructions.  
For dog breed, it is here:  
https://www.kaggle.com/c/dog-breed-identification#evaluation  

#### Submission File
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
ID, probabilities of each of the dog breeds  
```text
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

In our object, there is: 
```python
data.classes
```
```python
data.test_ds.fnames
```
[RS:  add notes later]

---
## CNN Behind the Scenes

* [A Visual and Intuitive Understanding of Deep Learning](https://www.youtube.com/embed/Oqm9vsf_hvU?autoplay=1&feature=oembed&wmode=opaque) Otavio Good of Google, AI Conf SF Sep 2017
* [mnist data](http://yann.lecun.com/exdb/mnist/) - we'll look at this dataset in more detail later

* [spreadsheets for example on CNN](https://github.com/fastai/fastai/tree/master/courses/dl1/excel)
* [conv-example (download spreadsheet from GH](https://github.com/fastai/fastai/blob/master/courses/dl1/excel/conv-example.xlsx)
   - every pixel is a number between 0 and 1.  (or sometimes 0 to 255)
   - by the time we get to PyTorch, we convert it to floats, so a number between 0 and 1
   - when Jeremey says activation, he means **a number**, a number that is calculated by taking some numbers from the input and applying some kind of linear operation, in this case a convolutional kernel, to calculate an output
   - max (0, x) --> Rectified Linear Unit = ReLU
   - 
   
#### Filter / Kernel   
top edge filter:  
```text
1 1 1
0 0 0
-1 -1 -1
``` 
left edge filter:
```text
1 0 -1
1 0 -1
1 0 -1
```
- PyTorch does not store as two separate 9-digit arrays.  It stores it as a **tensor**.  A tensor is an array with more dimensions.  (can also call it an array).  Tensor allows us to stack each of these filters together
- filter and kernel means the same thing here, a 3x3 matrix or slice of 3-dimensional tensor 
- each filter has created a **layer**, a hidden layer

Next filter will contain 2 of these kernels: 
**filter 1/2**  
```text
0.5 0.3 0.3
0.9 -0.5 0
0.8 0 -0.7
```
```text
0.8 0.4 -0.5
0.6 0.4 0.3
0.3 0.7 0.6
```
- shouldn't think of this as **2 3x3 kernels**, but 1 **2x3x3 kernel**
- over time, you want to start getting comfortable with idea of higher dimensional linear combinations
- conceptually, just stack it in your mind
- Jeffrey Hinton, in his 2012 neural nets Coursera class, has tips on how computer scientists deal with high dimensional space
- an **architecture** means how big is your kernel at layer 1, how many filters are in your kernel at layer 1
- an **activation** is an output from input times the kernel
- we give names to layers:  conv1, conv2, etc
- **maxpool** a 2x2 maxpooling will **half** the resolution, both height and width (over non-overlapping cells)
- **activation** take every single output from maxpool, and give them a **weight**, multiply them, and we get a **sum product**
- this is called a **fully connected layer**
- `01:03`:  architectures that make haeavy use of fully convolutional layers can have a lot of weights that take a lot of time; they may have trouble with overfitting and they can be slow
- we'll look at **VGG** architecture, has up to 19 layers.  first successful deep architecture.  VGG contains a fully connected layer.  Has 4,096 activations connected to a hidden layer with 4,096.  300 million weights of which 250 million are within fully connected layers
- resnet and resnext do not have a lot of fully connected layers behind the scenes
- R, G, B:  
- Kaggle iceberg competition, satellite with 2 channels

`01:08:45` after break  
What happens next?  After fully connected layer?  
In practice, if we want to calculate which of the 10 digits we're looking at, the single digit is not enough.
- We would have 10 sets of fully connected weights.
- last layer has no ReLU, so we can have negatives
- 

#### Logarithms

ln(x/y) = ln(x) - ln(y)

ln(x) = y   ----> e^y = x  

log_a(b) = ln(b) - ln(a)  

#### Planet Competition

Note:  Since images are matrices of numbers, and if the image looks washed-out, you can multiple it by a number, say 1.4:  
```python
plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4);
```
- these images are not at all like ImageNet
- vast majority of images we will use involving CNN will **not** be like ImageNet:  medical imaging, classifying different kinds of steel tube, figuring out if a weld will break or now, satellite images, etc.
- good to experiment with planet competition
- planet image data starts off 256 x 256
- resize it to 64 (wouldn't do this for cats / dogs because we start off with pretrained ImageNet data, it starts off nearly perfect.  most ImageNet models are trained on 224 x 224, or 299 x 299)
- There's nothing like satellite images ImageNet; some layers are helpful (finding edges, textures, repeating patterns

#### Training Planet Competition Data
- start with `sz = 64`
- 
- grab some data
- built model
- found out what learning rate to use
- needed to fit last layer before it started out flattening out
- then unfreeze:  `learn.unfreeze()`
- `learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)`
- save model:  `learn.save(f'{sz})`
- train for a while
- increase size:  `sz=128` (double the size)
- xxx
- End with TTA:  
   - `tta = learn.TTA()`
   - `f2(*tta)`
- used `metrics=[f2]` rather than `metrics=[accuracy]` 
- **f2** particular way of weighing false positives and false negatives


Note:  inputs * weights = activations  

#### Differential Learning Rates
```bash 
lr = 0.2
learn.fit(lr, 3, cycle_len = 1, cycle_mult = 2)
```
- there is a concept called **layer groups**
- type `learn.summary()` to get detailed info on layers

#### Sigmoid

---
## Structured Data
- Unstructured data:  images, audio, natural language text
- Structured data:  profit/loss statement, data in a spreadsheet, info about FB user, each column is structurally quite different (sex, zip code)
- structured data is what makes the world go around, though it is not presented at fancy conferences

### Grocery Store Competition
#### Current Ecuador grocery store competition
* [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
* this is current, so we won't work on it as a group

#### Former German grocery store competition
- Kaggle competition:  [Rossman Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
- Jupyter notebook:  [lesson3-rossman.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)

**Reminder:** **update repo** `git pull` and from time to time, **update conda**
```bash
git pull
conda env update
conda update --all 
```

### Pandas
recommended book:  [Python for Data Analysis](http://wesmckinney.com/pages/book.html) by Wes McKinney, 2nd Edition  
- 2nd edition release Sep 2017
- author of Pandas
- covers numpy, scipy, matplotlib, scikit-learn, iPython, Jupyter

### 
- structured data is generally shared as a csv file
- To get the Rossman data:  On AWS, we can do `wget http://files.fast.ai/part2/lesson14/rossman.tgz` 
- code tha tis in the notebook is from 3rd place winner:  [lesson3-rossman.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)  
- 

### [What is Feather?](http://blog.cloudera.com/blog/2016/03/feather-a-fast-on-disk-format-for-data-frames-for-r-and-python-powered-by-apache-arrow/)  
Feather is a fast, lightweight, and easy-to-use binary file format for storing data frames. It has a few specific design goals:

Lightweight, minimal API: make pushing data frames in and out of memory as simple as possible
Language agnostic: Feather files are the same whether written by Python or R code. Other languages can read and write Feather files, too.
High read and write performance. When possible, Feather operations should be bound by local disk performance.

## Next Week
- we'll go through steps in the lesson3-ross.ipynb notebook
- learn to split columns in 2 types:  categorical and continuous
   - Example:  Store ID 1 and 2 are not numerically related to each other, they are categories.  Day of Week also categorical.  Will do 1-hot encoding
   - distance of competitor is continuous
- try to enter as many kaggle competitions as possible

   
   
















