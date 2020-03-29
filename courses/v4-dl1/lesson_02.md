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

- "The loss function is used to optimize your model. (...) A metric is used to judge the performance of your model."
- model zoo: look for pretrained models
- bing image search: 7 days of high quota for free; limit to 3 transactions per second
