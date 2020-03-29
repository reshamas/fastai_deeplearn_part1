# Lesson 2
- Live:  24-Mar-2020
- Time: 6:30 to 9pm PST  (9:30pm to midnight EST)

- 9:30pm  250 viewing
- 10:00pm 314 viewing

## Homework
- [Lesson 2 Homework] ()

- [ ] read blog: [](https://www.fast.ai/2016/12/29/uses-of-ai/)
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 
- [ ] 

## Notes
- [fastai/fastbook](https://github.com/fastai/fastbook)
  - full notebooks that contain text of O'Reilly book
- [fastai/course-v4](https://github.com/fastai/course-v4) 
  - same notebooks with prose stripped away
  - do practice coding here

```python
from fastai2.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```
- `def is_cat(x): return x[0].isupper()` returns something that is True or False
- `dls` = data loaders
- `valid_pct=0.2` creates a validation set  (default is 20% of data is set to validation set)
- A **regression model** is one which attempts to predict one or more numeric quantities such as temperature or location.

### learner
- `learn = cnn_learner(dls, resnet34, metrics=error_rate)`
- `resnet` is architecture; `34` is number of layers in that architecture
- **epoch** is when you look at every single image in the dataset once
- `metrics=error_rate` the percent of validation set are being incorrectly classified by model
- `metrics=accuracy` is 1 - error_rate 
- **loss function** is a performance measurement
  - need a loss function where if you change the parameters by just a little bit up or just a little bit down you can see if the loss gets a little bit better or a little bit worse and it turns out that error rate and accuracy doesn't tell you that at all
- loss and metric are closely related, but metric is what you care about
- loss:  is what computer is using as measurement of performance to decide howto update your parameters
- **measuring overfitting** 
- "The loss function is used to optimize your model. (...) A metric is used to judge the performance of your model."
- model zoo: look for pretrained models
- bing image search: 7 days of high quota for free; limit to 3 transactions per second
- **test set** third set: not used for training or metrics (used on Kaggle) 
- training loss vs validation loss:  

