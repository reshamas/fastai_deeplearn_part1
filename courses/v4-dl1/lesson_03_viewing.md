# Lesson 3
- Live:  31-Mar-2020
- Time: 6:30 to 9pm PST  (9:30pm to midnight EST)

- 9:30pm  144 viewing
- 9:45pm  263 viewing
- 10:00pm  viewing

## Homework
- [Lesson 3 Homework] ()

- [ ] read blog: [](https://www.fast.ai/2016/12/29/uses-of-ai/)
- [ ] create your own application


## Notes
- [fastai/fastbook](https://github.com/fastai/fastbook)
  - full notebooks that contain text of O'Reilly book
- [fastai/course-v4](https://github.com/fastai/course-v4) 
  - same notebooks with prose stripped away
  - do practice coding here

## 
- using notebook:  https://github.com/fastai/fastbook/blob/master/02_production.ipynb
- look at getting model into production
- `DataBlock` API
```python
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.3, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
```

## Data Augmentation
- default: it grabs the center of image
- `.new`: creates a new DataBlock object
```python
bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```
- `ResizeMethod.Pad` adds black bars to side, avoids squishing image
- `pad_mode='zeros'` can have `pad_mode='reflect'`
`bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))`
- `ResizeMethod.Squish` most efficient
- `tem_tfms=RandomResizedCrop` most popular one; `min_scale=0.3` pick 30% of pixels of orig image each time
`bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))`

- Item transforms vs Batch transforms
```python
bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)
```
- fastai will avoid doing data augmentation on the validation dataset
- show name of cateogories:
```python
learn_inf.dls.vocab
```
```bash
(#3) ['black','grizzly','teddy']
```

## Making a GUI; web app for predictions (25:00)
- `!pip install voila`
- can use binder for making it publicly available

### *out of domain* data (domain shift)

### Python broadcasting

## MNIST: baseline + calculating gradient
- notebook:  https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb



