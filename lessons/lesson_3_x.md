# Lesson 3
(13-Nov-2017, live)

[Lesson 3 live stream](https://www.youtube.com/watch?v=9C06ZPF8Uuc&feature=youtu.be) 

[Wiki: Lesson 3](http://forums.fast.ai/t/wiki-lesson-3/7809)  

## Notebooks Used
* dog breed
* planet:  [lesson2-image_models.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson2-image_models.ipynb)  

## Notes

### Quick Dogs vs Cats
`precompute=True` when we use precomputed activations, data augmentation does not work.  Because `precompute=True` is using the cached, non-augmented activations.  

`learn.unfreeze()`    

`bn.freeze`

`learn.bn_freeze` If you're using a bigger, deeper model like resnet50 or resnext101, on a dataset that is very similar to ImageNet; This line should be aded.  This causes the batch normalization moving averages to not be updated.


`TTA` Test Time Augmentation - use to ensure we get the best predictions we can.  

fastai library sits on top of PyTorch.  

[`keras_lesson1.ipynb`](https://github.com/fastai/fastai/blob/master/courses/dl1/keras_lesson1.ipynb)  
