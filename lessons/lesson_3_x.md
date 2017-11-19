# Lesson 3:  xxx
(13-Nov-2017, live)

[Lesson 3 live stream](https://www.youtube.com/watch?v=9C06ZPF8Uuc&feature=youtu.be) 

[Wiki: Lesson 3](http://forums.fast.ai/t/wiki-lesson-3/7809)  

## Notebooks Used
* dog breed
* planet:  [lesson2-image_models.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson2-image_models.ipynb)  

---

## Resources

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

## Learning How to Download Data
- Kaggle
- other places on internet

### Kaggle CLI (command line interface)
Instructions on using `kaggle-cli`:  [kaggle_cli.md](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/kaggle_cli.md)  
Kaggle Competition:  [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

### Chrome CurlWget
Instructions on using Chrome CurlWget:  [chrome_curlwget](https://github.com/reshamas/fastai_deeplearn_part1/blob/master/tools/chrome_curlwget.md)

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
This code imports all the fastai libraries:  
```python
from fastai.conv_learner import *
PATH = "data/dogscats/"
sz=224; bs=48
```

`precompute=True` when we use precomputed activations, data augmentation does not work.  Because `precompute=True` is using the cached, non-augmented activations.  

`learn.unfreeze()`    

`bn.freeze`

`learn.bn_freeze` If you're using a bigger, deeper model like resnet50 or resnext101, on a dataset that is very similar to ImageNet; This line should be aded.  This causes the batch normalization moving averages to not be updated.


`TTA` Test Time Augmentation - use to ensure we get the best predictions we can.  

fastai library sits on top of PyTorch.  

[`keras_lesson1.ipynb`](https://github.com/fastai/fastai/blob/master/courses/dl1/keras_lesson1.ipynb)  
