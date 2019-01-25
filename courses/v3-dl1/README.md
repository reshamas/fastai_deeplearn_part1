
## Course Videos
https://course.fast.ai/videos


## Get to Jupyter Notebook
- Go to localhost (run Jupyter Notebook):  
http://localhost:8080/tree

## Important Links
- [Google Cloud Platform](http://course-v3.fast.ai/start_gcp.html)
  - [GCP: update fastai, conda & packages](http://course-v3.fast.ai/start_gcp.html#step-4-access-fastai-materials-and-update-packages)


[PyTorch Forums](https://discuss.pytorch.org)

## Model Tuning Advice

no graph for learning rate finder:  means learning rate is too small

### Seed for validation dataset
```python
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4)
```
This means that every time I run this code, I will get the same validation set.

### If errors are too high
#### example of problem
```bash
Total time: 00:13
epoch  train_loss  valid_loss  error_rate       
1      12.220007   1144188288.000000  0.765957    (00:13)
```

#### example of solution
```python
#learn.fit_one_cycle(6, max_lr=0.5)
#learn.fit_one_cycle(6, max_lr=0.25)
#learn.fit_one_cycle(6, max_lr=0.05)
#learn.fit_one_cycle(6, max_lr=0.025)
#learn.fit_one_cycle(6, max_lr=0.01)
learn.fit_one_cycle(6, max_lr=0.001)
```

### LR finder plot is blank
#### 1.
```python
learn.recorder.plot()
# if plot is blank
learn.recorder.plot(skip_start=0, skip_end=0)
```

#### 2.  reduce batch size
- Reducing your batch size, in order to increase the number of batches.
```python
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4, bs=16)
```

You’re now overfitting. Try 10 epochs, then unfreeze, then 4 epochs.

