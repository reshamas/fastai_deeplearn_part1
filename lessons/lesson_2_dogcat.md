# Lesson 2:  CNN, dogs/cats 
(06-Nov-2017, live)

https://www.youtube.com/watch?v=JNxcznsrRb8&feature=youtu.be


[Wiki: Lesson 2](http://forums.fast.ai/t/wiki-lesson-2/7452)  

## Notebooks Used 
[fast.ai DL lesson1.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)  

---

 
---

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

  



