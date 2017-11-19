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


## SGDR (Stochastic Gradient Descent with Restarts)


### Confusion Matrix
* What was the actual truth?  Of the thousand actual cats, how many did we predict as cats?

### Review: easy steps to train a world-class image classifier
- Enable data augmentation, and precompute=True
- Use lr_find() to find highest learning rate where loss is still clearly improving
- Train last layer from precomputed activations for 1-2 epochs
- Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
- Unfreeze all layers
- Set earlier layers to 3x-10x lower learning rate than next higher layer
- Use lr_find() again
- Train full network with cycle_mult=2 until over-fitting

And more...  
- Use lr_find() to find highest learning rate where loss is still clearly improving
- Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
- Unfreeze all layers
- Set earlier layers to 3x-10x lower learning rate than next higher layer
- Train full network with cycle_mult=2 until over-fitting

  



