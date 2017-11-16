# Lesson 2:  CNN, dogs/cats 
(06-Nov-2017, live)

https://www.youtube.com/watch?v=4mwdySNmtYs  

[Wiki: Lesson 2](http://forums.fast.ai/t/wiki-lesson-2/7452)  

## Notebooks Used 
[fast.ai DL lesson1.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)  

---

`19:00` start of Lesson 2 

`01:33` to `01:45` break

`01:47` Teams info  
* what do you want to get out of deep learning
* share forum IDs
  
`01:55` back to lecture, meeting at USF  
* study groups

`01:57` back to DL lecture  * start here...

`02:42` discuss AWS fast.ai AMI  

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

  



