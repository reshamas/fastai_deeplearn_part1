# Lesson 2:  CNN, dogs/cats 
(06-Nov-2017, live)

https://www.youtube.com/watch?v=JNxcznsrRb8&feature=youtu.be


[Wiki: Lesson 2](http://forums.fast.ai/t/wiki-lesson-2/7452)  

## Notebooks Used 
[fast.ai DL lesson1.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)  

---

## Learning Rate
- how quickly will we zone in on the solution
- we take the gradient, which is how steep is it at this point, and we multiply it by some number, which is the running rate.
- if that number is small, we will get closer, slowly
- if we take a number too big, we could be far from our minimum
- if our loss is spinning off into infinity, most likely our learning rate is too high
- Wouldn't it be nice if there was a way to figure out the best learning rate?

```text
10^-1 = 0.10 = 1e-1
10^-2 = 0.001 = 1e-2
10^-3 = 0.001 = 1e-3
```

```python
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 3)
```
```text
Col 0:  Epoch Number
Col 1:  loss on training
Col 2:  loss on validation
Col 3:  accuracy
```
```
A Jupyter Widget
[ 0.       0.03597  0.01879  0.99365]                         
[ 1.       0.02605  0.01836  0.99365]                         
[ 2.       0.02189  0.0196   0.99316]
```
## precompute = True

## Data Augmentation
```python
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
```  
- another option:  `transforms_top_down`
- can also create custom transforms
- data augmentation is not exactly creating new data, but it's a different way of looking at it for the convolutional neural network
- 

## Unfreeze Layers

## Learning Rate Annealing

## TTA (Test Time Augmentation)

## fastai library
- open source
- sits on top of PyTorch
- PyTorch is fairly new; was not able to use Keras or TensorFlow
- PyTorch is not easy to use.
- So, created a library on top of PyTorch.
- PyTorch didn’t seem suitable for new deep learners
- With keras, code from last year’s course, is 2-3x longer, which means more opportunities for mistakes
- So, fastai built this library to make it easier to get state-of-the-art results
- Using this library made it so much more productive.
- were able to add in other papers 
- not only does fastai let us do things easier than other approach, it has more sophisticated stuff behind the scene
- fastai library has been released, open source
- behind the scenes, it is creating PyTorch models which can be exported
- if you're doing something on mobile, you'll need to use TensorFlow
- every year, the libraries that are available and the best change
- main thing to get out of this course is to get the **concepts**
  - learning rate
  - how to learning rate annealing
  - why differential learning rates are important
  - stochastic gradident descent with restart

Pyro - Uber's new release

## SGDR (Stochastic Gradient Descent with Restarts)


### Confusion Matrix
- simple way to look at the results of classification
* What was the actual truth?  Of the thousand actual cats, how many did we predict as cats?

## Review: easy steps to train a world-class image classifier
1.  Enable data augmentation, and precompute=True
2.  Use lr_find() to find highest learning rate where loss is still clearly improving
3.  Train last layer from precomputed activations for 1-2 epochs
4.  Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
5.  Unfreeze all layers
6.  Set earlier layers to 3x-10x lower learning rate than next higher layer
7.  Use lr_find() again
8.  Train full network with cycle_mult=2 until over-fitting

And more...  
- Use lr_find() to find highest learning rate where loss is still clearly improving
- Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
- Unfreeze all layers
- Set earlier layers to 3x-10x lower learning rate than next higher layer
- Train full network with cycle_mult=2 until over-fitting

## Dataset 2:  Dog Breed Competition
* can set `sz=64`, use small size photo in beginning to get model running, and then increase the size
* most ImageNet models are trained on 224x224 or 299x299 sized images.  Images in that range will work well with these algorithms.
* **epoch** - number of passes thru the data
* **cycle** - however many epochs you say is in that cycle
* if **cycle** is 1, then cycle and epoch are the same
* starting training on a few epochs with small size `sz=224` and then pass in a larger size of images and continue training.  This is another way to get state-of-the-art results.  Increase size to 299. If I overfit with 224 size, then I'm not overfitting with 299.  This method is an effective way to avoid overfitting.
* 

#### Note
* the best way to deal with unbalanced data is to make copies of the rare cases

### precompute=True
* started with a pre-trained network; found activations with rich features; then we add a couple of layers at the end, which start off random
* with freeze (frozen by default) and `precompute=True`, all we are learning is the couple of layers we've added
* with `precompute=True`, we actually precalculate how much does this image have the features such as eyeballs, face, etc.
* **data augmentation** doesn't do anything with precompute=True because we're actually showing the same exact activations every time.
* we can then set `precompute=False`, which means it is still only training the last couple of layers, but **data augmentation** is now working because it is going through and re-calculating all the activations from scratch 
* finally, when we unfreeze, we can go back and change the earlier convolutional filters
* having precompute=True initially makes it faster, 10x faster.  It doesn't impact the accuracy.  It's just a shortcut.

* if you're showing the algorithm less images each time, then it is calculating the gradient with less images, and is less accurate
* if making batch size smaller, making algorithm more volatile; impacts the optimal learning rate.  
* if you're changing the batch size by much, can reduce the learning rate by a bit.







