# Lesson 9: Multi-object detection
(26-Mar-2018, live)  
 
- [Wiki: Part 2 / Lesson 9](http://forums.fast.ai/t/part-2-lesson-9-wiki/14028)
- [Lesson 9 video](https://www.youtube.com/watch?v=0frKXR-2PBY) 
  - video length:  2:05:33
- http://course.fast.ai/lessons/lesson9.html
- Notebook:  
   * [pascal.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/pascal.ipynb)

---

## Start Class
- today we will continue working on object detection, which means that for every object in a photo with 1 of 20 classes...
- we are going to figure out what the object is, and what its bounding box is such that we can apply that model to a new dataset with unlabeled data and add those labels to it
- general approach is to start simple and gradually make it more complicated; we started last week with a simple classifier, 3 lines of code, 
- we make it slightly more complex to turn it into a bounding box without a classifier
- today, we will put those two pieces together to make a classifier plus a bounding box 
- all of these are for a single object, the largest object in the image, and from there we will build up something which is closer to our final goal
- you should go back and make sure you understand all the concepts from last week

## Things to Know from Last Week
- Pathlib; JSON
- dictionary comprehensions
- `defaultdict`
- knowing how to jump around fastai source code is important
- lambda functions --> they come up everywhere
- matplotlib API
- Bounding box coordinates
- Custom head; bounding box regression --> will come up in every lesson
<br>
<img src="../../images/lesson_09/lesson9_know_these1.png" align="center"   >   
<br>
<img src="../../images/lesson_09/lesson9_know_these2.png" align="center">   


## What You Should Know from Part 1 of Course
- How to view model inputs from a DataLoader
- How to view model outputs
- need to know this to learn how to debug models

<img src="../../images/lesson_09/lesson9_data_loader.png" align="center"  height="300" width="500" >   


## Today's Lesson (`3:00`)
- we were working through Pascal notebook
- we had gone over creating bounding box over largest object without a classifer
- Jeremy had skipped over augmentations last week

## Augmentations
- data augmentations of `y`, the dependent variable
- before proceeding, Jeremy wants to talk about something awkward `continuous=True` in the statement 
```python
tfmstfms  ==  tfms_from_modeltfms_fro (f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms,
   bs=bs, continuous=True, val_idxs=val_idxs)
```
- this makes no sense whatsoever because a classifier is a dependent variable as categorical or binomial as opposed to regression which is anything where the dependent variable is continuous
- AND YET, this parameter here, `continuous=True`, says that the dependent variable is continuous
- so this claims to be creating data for a classifier where the dependent is continous;
- this is the kind of awkward rough edge that you see when we're kind of at the edge of the fastai code that is not quite solidified yet
- probably by the time you watch this in the MOOC, it will be sorted out before you even regress it
- Jeremy wanted to point out this issue; sometimes people get confused

### Let's Create Some Data Augmentations (`4:40`)
- normally when we create data augmentations, we tend to type in "transform side on" or "transforms top down"
- file:  [transforms.py](https://github.com/fastai/fastai/blob/master/fastai/transforms.py)
```python
transforms_basic    = [RandomRotate(10), RandomLighting(0.05, 0.05)]
transforms_side_on  = transforms_basic + [RandomFlip()]
transforms_top_down = transforms_basic + [RandomDihedral()]
```
- but, if you look in the transforms.py module, you will see that they are simply defined as a list
- transforms_basic --> is 10 degrees rotation plus 0.05 brightness and contrast
- transforms_side_on --> adds to the basic transform and random horizontal flips
- transforms_top_down --> adds to the basic transform and adds random dihedral group of symmetry which is flips which basically means every possible 90 degree rotation 
- these are ones created by Jeremy; you can always create your own augmentations
```python
augs = [RandomFlip(), 
        RandomRotate(30),
        RandomLighting(0.1,0.1)]
```
- if you are not sure which augmentations are there, you can check the source code, or if you start typing "random" and tab for auto-complete, you will see the options
- let's see what happens if we create some data augmentations
- then, create a model data object; let's go through and re-run the iterator a bunch of times and we all do two things
- we print out the bounding boxes and so you can see the value box is the same each time and we will also draw the pictures 

<br>
<img src="../../images/lesson_09/lesson9_bbox.png" align="center">   
<br>

- you will see, this lady in the photo has flipped image, contrast is changing, as we would expect
- but you'll see the bounding box is the same each time in the photo and not moving AND is in the WRONG spot
- this is the problem with data augmentation when your dependent variable is pixel values or is in some way connected to your independent variable; the two need to be augmented together
- you can see from the printout that the numbers are bigger than 224 (bounding box coords), but these images are size 224
- the images are not being scaled or cropped
- you can see our dependent variable needs to go through all the same geometric transformations w/o independent variables
- to do that, every transformation has an optional transform `Y` parameter, it takes a transform type `TfmType` 
- `TfmType` has a few options, all of which will be covered in this course
- reminder, hit tab after the `TfmType.` to get options
```python
augsaugs  ==  [[RandomFlipRandomFl (tfm_y=TfmType.COORD),
        RandomRotate(30, tfm_y=TfmType.COORD),
        RandomLighting(0.1,0.1, tfm_y=TfmType.COORD)]
```
- (`7:33`) the `.COORD` option says that the y values represent coordinates, in this case, bounding box coordinates
- therefore, if you flip, you need to change the coordinate to represent that flip or if you rotate, change the coordinate to represent that rotation
- so, I can add transform type coord `tfm_y=TfmType.COORD` to all of my augmentations
- I also have to add the exact same thing to my transforms from model function because that is the thing that does the cropping and/or zooming and padding and/or resizing, and all of those things need to happen to the dependent variable as well
- so, if we add all of those together and rerun this, you'll see the bounding box changes each time with each augmented image
- you'll see the bounding box is in the right spot now
- you'll see sometimes it looks a little odd and the problem is this is just a constraint that the information we have
- the bounding box does not tell us that actually her head isn't way over in the top left corner
- but, actually, if you do a 30 degree rotation if her head was over in the top left corner, then the new bounding box would go really high
- so, this is the correct bounding box based on the information it has available which is to say this is how it might have been
- so, you have to be careful of not doing too high a rotations with bounding boxes because there is not enough information for them to stay totally accurate
- just a fundamental limitation of the information we are given
- if we were doing polygons or segmentations, or whatever else, we wouldn't have this problem
- so, Jeremy is going to a **maximum of 3 degrees** rotation to avoid that problem
- Jeremy is also only going to rotate half the time `p=0.5` 
- here's the set of transformations Jeremy is using:
```python
tfm_ytfm_y  ==  TfmTypeTfmType..COORDCOORD
 augsaugs  ==  [[RandomFlipRandomFl (tfm_y=tfm_y),
        RandomRotate(3, p=0.5, tfm_y=tfm_y),
        RandomLighting(0.05,0.05, tfm_y=tfm_y)]
```

### Custom Head Idea (`09:35`)
- we briefly looked at this custom head idea but basically if you look at `learn.summary()` it does something cool, like runs a small batch of data through a model and prints out how big it is at every layer
- we can see at the end of the convolutional section before we flatten, it is `512 x 7 x 7`
- 512x7x7 = 25,088, rank 3 tensor, if we flatten it out into a single rank=1 tensor (to a vector), it will be 25088 long
- that's why we have this linear:  `nn.Linear(25088, 4)`, there are 4 bounding boxes
```python
learn.summary()
```
```python
Out[170]:
OrderedDict([('Conv2d-1',
              OrderedDict([('input_shape', [-1, 3, 224, 224]),
                           ('output_shape', [-1, 64, 112, 112]),
                           ('trainable', False),
                           ('nb_params', 9408)])),
             ('BatchNorm2d-2',
              OrderedDict([('input_shape', [-1, 64, 112, 112]),
                           ('output_shape', [-1, 64, 112, 112]),
                           ('trainable', False),
                           ('nb_params', 128)])),
             ('ReLU-3',
              OrderedDict([('input_shape', [-1, 64, 112, 112]),
                           ('output_shape', [-1, 64, 112, 112]),
                           ('nb_params', 0)])),
```
- stick that on top of a pretrained resnet
- train it for a while 
```python
head_reg4head_reg4 = nn.Sequential(Flatten(), nn.Linear(25088,4))
learn = ConvLearner.pretrained(f_model, md, custom_head=head_reg4)
learn.opt_fn = optim.Adam
learn.crit = nn.L1Loss()
```

## Single Object Detection
- let's now put those two pieces together so we get something that **classifies** and does **bounding boxes**
- there are 3 things that we need to do to train a neural network ever:
  1.  we need to provide data
  2.  we need to pick some kind of architecture
  3.  loss function - anything that says a lower number here is a better network using this data in this architecture
- we are going to need to create the above 3 things for our classification plus bounding box regression
- that means we need a model data object which has as independent, the images, and as the dependent, I want to have a tuple, first one of the tuples should be the bounding box coordinates, the second element of the tuple should be a class
- there's lots of different ways you can do this
- the particularly lazy and convenient way that Jeremy came up with was to create two mobile data objects, representing the two different dependent variables I want; so one with the bounding box coordinates and one with the classes
- just using the csvs we went over before
- now I am going to merge them together
- so I create a new dataset class
- a dataset class is anything which has a length and and index service or something that lets you use it in square brackets like lists
- in this case, I can have a constructor which takes an existing dataset, so that's going to have both an independent and dependent, and the second dependent that I want is 
- the length, that is obviously just the length of the dataset, the first dataset
- and `getitem` is grab the `X` and the `y` from the dataset that I passed in and return that X and that Y and the i'th value of the second 
- so, there's a dataset that basically adds in 
- as I said, there are lots of ways to do this
```python
class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2): self.ds,self.y2 = ds,y2
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        x,y = self.ds[i]
        return (x, (y,self.y2[i]))
```
- this is convenient because now I can create a training dataset and a validation dataset
```python
trn_ds2 = ConcatLblDataset(md.trn_ds, md2.trn_y)
val_ds2 = ConcatLblDataset(md.val_ds, md2.val_y)
```
- here's an example
- you can see it's got a couple of the bounding box coordinates in the class
- we can then take the existing training and validation data loaders now so you replace their datasets with these and unknown
```python
val_ds2val_ds2[[00][][11]]
```
```python
(array([   0.,   49.,  205.,  180.], dtype=float32), 14)
```
- we can now test it by grabbing a mini batch of data and checking it so it's ok

### Architecture
- we have the data, now we need an architecture
- the architecture we will use will be the same ones used for the classifier and bounding box regression, but we're going to combine them
- if there are "C" classes, then the number of activations we need in the final layer is:  4+C (4 for coordinates, C probabilities one per class)
- this is the final layer, a linear layer that has 4 plus len of categories:  `nn.Linear(256, 4+len(cats)),`
- the first layer is:  `Flatten()`
- we could just join those together, but, in general, I want my custom head to hopefully be capable of solving the problem I give it on its own, if the pretrained backbone is connected to is appropriate 
- in this case, I am trying to do quite a bit here (in function `head_reg4`), two things: classifer and bounding box regression
- so, just a single linear layer doesn't sound like enough, so Jeremy puts in a second linear layer: `nn.Linear(25088, 256),`
- you can see we do:  nn.ReLU, nn.Dropout, nn.Linear(25088, 256), nn.ReLU, nn.BatchNorm1d, nn.Dropout, nn.Linear
- if you are wondering why there is no `BatchNorm1d1` after the first ReLU, Jeremy checked the ResNet backbone; it already has a BatchNorm as its final layer
- so, this is nearly the same custom head as before, it it's just got 2 linear layers rather than one, and the appropriate non-linearities (such as ReLU)
- that's piece 2; we have got data, we've got architecture, now we need a **Loss Function**

<br>
<img src="../../images/lesson_09/lesson9_archit.png" align="center">   
<br>

### Loss Function (`15:30`)
- the loss function needs to look at the **"4+C"** activations and decide, "are they good?"
- are these numbers accurately reflecting the position and the class of the largest object in this image
- we know how to do that 
- for the first 4, we use the **L1 loss**, just like we did in the bounding box regression before
- remember, **L1 loss** is like **Mean Squared Error**, rather than *sum of absolute values* / *sum of absolute values*
- the rest of activations, we can use **cross-entropy loss**
- let's go ahead and do that; we're going to create something called "detection loss"
- loss functions always take an **input** and a **target** --> that's what PyTorch calls them
- input (activations; target (ground truth)
- remember that our custom data set returns a tuple containing the bounding box coordinate and the classes of the target
- so we can destructure that, 
- `bb_i,c_i = input[:, :4], input[:, 4:]` bounding boxes and the input are simply the first 4 elements of the input and the 4 elements onward
- remember we've also got a batch dimension 
- for the bounding boxes, we know they are going to be between 0 and 224 coordinates, because that's how big the images are
- so, let's grab a sigmoid and force it betweeen 0 and 1, and multiply it by 224, and that's just helping our neural net get close to what we know it has to be 
- Q:  as a general rule, is it better to put batch norm before or after ReLU?
- A:  Jeremy suggests that you should put it after a ReLU because batch norm is meant to move towards a 0/ 1 random variable; if you put it after, you're truncating it at 0, so there's no way to create negative numbers, but if you put reLU and then batch norm, it does have that ability
- that way of doing it gives slightly better results; having said that, it's not too big a deal either way, and you'll see during this part of the course most of the time, Jeremy goes ReLU and then batch norm, but sometimes it is batch norm and ReLU if JH is being consistent with a paper
- `bb_i = F.sigmoid(bb_i)*224` so this is to force our data into the right range; if you can do stuff like that, it is easier to train
- Rachel's question:  what's the intuition behind using dropout with p=0.5 after a batch norm? Doesn't batch norm already do a good job of regularizing?
- JH answer:  batch norm does an okay job of regularizing, but if you think back to part 1, we have a list of things to do to avoid overfitting and adding batch norm is one of them, as is data augmentation, but it's perfectly possible that you'll still be ok.
```python
def detn_loss(input, target):
    bb_t,c_t = target
    bb_i,c_i = input[:, :4], input[:, 4:]
    bb_i = F.sigmoid(bb_i)*224
    # I looked at these quantities separately first then picked a multiplier
    #   to make them approximately equal
    return F.l1_loss(bb_i, bb_t) + F.cross_entropy(c_i, c_t)*20

def detn_l1(input, target):
    bb_t,_ = target
    bb_i = input[:, :4]
    bb_i = F.sigmoid(bb_i)*224
    return F.l1_loss(V(bb_i),V(bb_t)).data

def detn_acc(input, target):
    _,c_t = target
    c_i = input[:, 4:]
    return accuracy(c_i, c_t)

learn.crit = detn_loss
learn.metrics = [detn_acc, detn_l1]
```

---
video
`13:36`
