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

### transfer learning
- `learn.fine_tune(1)`
- fine tuning is a transfer learning technique where the parameters of a pretrained model are trained for additional epochs for a different task
- update later layers of a pretrained model
- techniques for computer visio can be used for **sound**, turn sounds into pictures by showing their frequencies over time, which results in an image
- example of Splunk:  using computer vision to create an anti-fraud model
  - [Splunk and Tensorflow for Security: Catching the Fraudster with Behavior Biometrics](https://www.splunk.com/en_us/blog/security/deep-learning-with-splunk-and-tensorflow-for-security-catching-the-fraudster-in-neural-networks-with-behavioral-biometrics.html)
  - it worked so well Splunk patented a new model
- example:  turn viruses into pictures
 
### Metrics
- loss function: a measure of performance, when we adjust our hyperparameters up and down, how it changes
  - a measure of performance that the algorithm uses to try to make the algorithm better.  
  - as you change the parameters a bit, the loss should always change
- error:  is one kind of metric
- metric:  
- parameter:  things which change what the model or architecture does.  start with neural network which is infinitely flexible. 
  - pass in:  numbers that represent input (pixels)
  - pass in:  learned parameters
  - numbers which change what the model does
- hyperparameter:  choices about which numbers we pass to model for fitting function

- filters: are they independent? if they are fine-tuned, do they lose some information?
  - **catastrophic forgetting** in literature
  - if you want to fine tune something which is good at new task and continues to be good at previous task, need to continue to put in examples of previous task as well
- pacing of course:  we will cover what we can in 7 lessons;  whole book will be covered in 2 or 3 courses; book is 500 pages
  - so will be 14 or 21 lessons to get through whole book
- part of course:  putting things into production
  - what are capabilities and limitations?
  - first 2 or 3 lessons of course are for coders and everyone; what are practical things you need to know?
  - what is deep learning good at, at the moment?
- pre-trained weights
  - there are a lot of pretrained models
  - 

## Deep Learning Applications
- Vision:  detection, classification
- Text:  classification, conversation
- Tabular: high cardinality; GPU (rapids)
- Recsys:  prediction NE (<>) recommendation
- Multi-modal: labeling, captioning, human in the loop
- Other: NLP -> protein




