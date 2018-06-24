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
- 
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
