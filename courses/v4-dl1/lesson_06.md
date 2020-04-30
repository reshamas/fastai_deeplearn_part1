# Lesson 6
- Live:  21-Apr-2020
- Time: 6:30 to 9pm PST  (9:30pm to midnight EST)
- 10:30pm 185 watching

## Homework
- [Lesson 4] ()
- [ ] 
- [ ] 

## Notes
- [fastai/fastbook](https://github.com/fastai/fastbook)
  - full notebooks that contain text of O'Reilly book
- [fastai/course-v4](https://github.com/fastai/course-v4) 
  - same notebooks with prose stripped away
  - do practice coding here
  
## Topics
- pet breeds; multiple classification
- good learning rate finder questions and answers

## Computer Vision Problem: Pet Breed

### Discriminative Learning Rates
- Notebook:  https://github.com/fastai/course-v4/blob/master/nbs/05_pet_breeds.ipynb
- unfreezing and transfer learning
>what we would
really like is to have a small learning
rate for the early layers and a bigger
learning rate for the later layers
- slicing
```python
learn.fit_one_cycle(6, lr_max=1e-5)
```
#### our own version of fine-tuning here
```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))
```
#### how do you make it better now?
- 5.4% error on 37 categories is pretty good (for pet breed data)
- can use a deeper architecture
- `Cuda runtime error: out of memory` is out of memory on your GPU
  - restart notebook
  - can use less precise numbers to save memory
```python
from fastai2.callback.fp16 import *
learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs=3)
```
- increasing number of layers (or more complex architecture) doesn't always improve the error rate
- requires experimentation
- trick:  use small models for as long as possible (to do cleaning and testing); then try bigger models because they will take longer
- "always assume you can do better [with error rate] because you never know"

## Multi-label Classification
- notebook:  https://github.com/fastai/course-v4/blob/master/nbs/06_multicat.ipynb
- determining multiple labels per image (Ex: contains car, bike person, etc)
- dataset:  PASCAL
  - http://host.robots.ox.ac.uk/pascal/VOC/
  - https://gluon-cv.mxnet.io/build/examples_datasets/pascal_voc.html


## Example
```python
a = list(enumerate(string.ascii_lowercase))
a[0], len(a)
```
```bash
((0, 'a'), 26)
```

## creating: **Datasets**, **Data Block** and **DataLoaders**
- serialization: means saving something
- best to use functions over lambda (because in Python, it doesn't save object created using lambda)
- one-hot encoding for multiple labels
- 
```python
def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train,valid

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y)

dsets = dblock.datasets(df)
dsets.train[0]
```
## path
```python
Path.BASE_PATH = None
path
```
```python
(path/'01').ls()
```
### Important to know
1. create a learner
2. grab a batch of data
3. pass it to the model
4. see the shape; recognize why the shape is
```python
learn = cnn_learner(dls, resnet18)
```
```python
x,y = dls.train.one_batch()
activs = learn.model(x)
activs.shape
```
>torch.Size([64, 20])





