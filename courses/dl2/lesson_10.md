# Lesson 10:  NLP Classification and Translation
(02-Apr-2018, live)  

### Part 2 / Lesson 10
- [Wiki Lesson 10](http://forums.fast.ai/t/part-2-lesson-10-wiki/14364)
- [Video Lesson 10](https://www.youtube.com/watch?v=h5Tz7gZT9Fo&feature=youtu.be) 
  - video length:  2:07:55
- http://course.fast.ai/lessons/lesson10.html
- Notebook:  
   * [imdb.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb)

---

## Review of Last Week
- if you are finding the material difficult, that's ok
- there are quite a few in-class fastai students who are doing this full-time and some of them are struggling with this material
- one of the reasons Jeremy had that content early on is we've got something to **cogitate** and think about and gradually work towards so by Lesson 14, you'll get a second crack at it

## Reading Papers (Research Journal Publications)
- not enough students are reading the papers!  The papers are the real ground truth
- a lot of people aren't reading the papers because they think they can't read the papers, but YOU ARE, because you are here

### Debugger
```python
pdb.set_trace()
```
### `00:14:20` NLP
- we've seen the idea of taking a pre-trained model, whip off some stuff from the top, replace it with new stuff, get it to do something similar
- we've dived a little bit deeper with that; with ConvLearner.pretrained, it had a standard way of sticking stuff on the top which does a particular thing, which was classification
- and then we learned, we can stick any PyTorch module at the end and have it do anything we like with a custom head
- suddenly you discover there are some interesting things we can do
- `00:15:35` YangLu said, what if we did a different kind of custom head? 
    - take original picture, rotate it, and then make our dependent variable the opposite of that rotation
    - and see if it can learn to un-rotate it
    - and this is a super useful thing, Google photos has an option to automatically rotate photos for you
    - as Yang Lu showed here, you could build that network right now by doing exactly the same as our previous lesson, but your custom head is one that spits out a single number, which is how much to rotate by and your dataset has a dependent variable which is how much did you rotate by?  And your dataset has a dependent variable, which is how much did you rotate by?
- so, you suddenly realize the idea of a backbone with a custom head, you can do almost anything you can think about
- Next, let's think about, how does that apply to NLP?


### Match the Equations to the Code
- [List of Mathematical Symbols](https://en.wikipedia.org/wiki/List_of_mathematical_symbols)

### Re-create things (graphs) you see in the papers
- "Try to recreate this chart, and make sure you understand what it's saying and why it matters"

### `00:16:40` We've moved from *torchtext* to *fastai.text*
#### NLP
- in the next lesson, we're going to go further and say, if NLP and computer vision lets you do the same basic ideas, then how do we combine the two? and we're going to learn about a model that we'll learn to find word structures from images, OR images from word structures, OR images from images!
- and that will form the basis, if you wanted to go further, of doing things from an image to a sentence, "image captioning", or going from a sentence to an image, which we will start to do "phrased image"
- so, from there, we'll go deeper into computer vision to think about what other kinds of things we can do with this idea of pre-trained network + custom head
- `00:17:45` we'll look at various kinds of image enhancement, like:
  - increasing the resolution of a low-res photo to guess what was missing
  - or adding artistic filters on top of photos
  - or changing photos of horses into photos of zebras and stuff like that 
- and finally, that will bring us all the way back to bounding boxes again
- to get there, we will first learn about segmentation, which is not just figuring out where the bounding box is, but figuring out what every single pixel in an image is part of
  - is this pixel part of a person, a car? 
- and then we will use that idea, idea of **U-Net** and it turns out this idea from unet, we can apply the idea of bounding boxes, which are called **feature pyramids** (everything has to have a different name in every slightly different area)
- and we'll use that to get very good results with bounding boxes
- that's our path from here, it all builds on each other, but take us into lots of different areas

### 4 Reasons Why..  *torchtext* to *fastai.text*
- no parallel processing
- hard to do simple things (like multi-label classification)
- no obvious way to save intermediate calculations
- somewhat convoluted API

#### `00:18:55` Section notes
- For NLP in Part 1, we relied on a pretty great library called torchtext, but as pretty great as it was, JH has 
found problems with it that are limiting, too problematice to keep using it
- as a lot of students complained on the forums, it's very slow because it's not doing parallel processing and doesn't remember what it did previously, so reruns it, does it all over from scratch
- and it's hard to do fairly simple things, a lot of students tried to the Kaggle Toxic Comment Challenge, which was a multi-label problem, and doing that with torchtext, JH eventually got it working, but it took him a week to hack away

### fastai.text
- `00:19:50` to fix all these problems, JH has created a new library called *fastai.text*
- fastai.text is a replacement for the combination of torchtext and fastai.nlp.  **Don't use fastai.nlp anymore** --> that is obsolete
  - it's slower, more confusing, less good in every way, but lot of overlaps, 
  - a lot of the classes & functions have the same names, that is intentional
- this is the non-torchtext version

## IMDb data `00:20:30`
- Notebook:  [imdb.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb)
- we'll work with the IMDb again, for those of you who have forgotten, go check out lesson 4, IMDb reviews
- this is a dataset of movie reviews, and you remember we used it to find out whether we might enjoy ?somebegeddon
- we are going to use the same dataset
- by default, it calls itself "aclImDB":  `data/aclImdb/')`  --> this is the raw dataset that you can download
- as you can see (there is no torchtext, and I'm not using fastai.nlp):
```python
from fastai.text import *
import html
```
- JH is using `Path` lib, as usual; we'll learn about the tags later
```python
BOS = 'xbos' # beginning-of-sentence tag
FLD = 'xfld' # data field tag

PATH = Path('data/aclImdb/')
```
- you'll remember, the basic path for **NLP** is we have to take sentences and turn them into numbers, and there are a couple of steps to get there
- at the moment, somewhat intentionally, fastai.text doesn't provide that many helper functions, it's really designed more to let you handle things in a fairly flexible way  
- as you can see here, I wrote something called `get_texts` which goes through each thing in "classes".  These are the 3 things they have in classes: negative, positive and unsupervised (stuff they haven't gotten around to labeling yet)
- JH goes thru each one of those classes, find every file in that folder with that name, and I open it up and read it and chuck it into the end of the array `texts`
- as you can see, with path lib, it's easy to grab stuff and read it in
- and then the label is whatever class I'm up to
- JH will go ahead and do it for the **train** bit and the **test** bit
```python
CLASSES = ['neg', 'pos', 'unsup']

def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

trn_texts,trn_labels = get_texts(PATH/'train')
val_texts,val_labels = get_texts(PATH/'test')
```
- so there are 75000 in train, 25000 in test (50,000 of the train are unsupervised); we won't actually be able to use them when we get to the classification piece
- JH actually finds this easier than the torchtext approach of having lots of layers and wrappers and stuff, because at the end, reading text files is not that hard

#### Sorting `00:23:20`
- one thing that is always a good idea is to sort things randomly
- it's useful to know this simple trick for sorting things randomly, particularly when you've got multiple things that you have to sort the same way
- in this case, you've got labels and texts
- 
```python
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))
```

```python
trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]
```

###





