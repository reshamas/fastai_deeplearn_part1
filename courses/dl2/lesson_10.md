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
- `np.random.permutation` if you give it an integer, it gives you a random list from 0 up to the number you give it (not including the number), in some random order, and so you can pass that in as an indexer.
```python
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))
```
- to give you a list that is sorted in that random order
- in this case, it will sort `trn_texts` and `trn_labels` in the same random way
- that's a useful little idiom to use
```python
trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]
```
- now, I've got my texts and labels sorted
- I can go ahead and create a dataframe from them
- Why am I doing this?  Because there is a somewhat standard approach starting to appear for text classication datasets, which is to have your training set as a csv file, 
  - with the labels first, and the text of the NLP document second   `col_names = ['labels', 'text']`
  - with a train.csv and a test.csv
  - and a file called `classes.txt` which just lists the classes
  - it is somewhat standard
  - in a reasonably recent academic paper, Yann LeCun and a team of researchers looked at quite a few datasets, and they used this format for all of them 
  - that's what JH has started using as well for his recent papers
```python
df_trndf_trn  ==  pdpd..DataFrameDataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)
```
- **if you put your data into this format, you'll find that the whole notebook will work on your dataset, every time!**
- so, rather than having a thousand different classes or formats or readers and writers or whatever; let's pick a standard format and your job is to put your data into that format which is a csv file
- **the csv files have no header, by default**
- you'll notice at the start, there are two different paths:
  - one was the classification path --> contains info we'll use to create the sentiment analysis model
  - other was the language model path (lm = language model) --> info to create language model
- when we create the `CLAS_PATH/'train.csv'`, we remove everything that has a label of 2 (which is "unsupervised")  
```python
CLAS_PATH=Path('data/imdb_clas/')
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH=Path('data/imdb_lm/')
LM_PATH.mkdir(exist_ok=True)
```
- so that means the data we use will have 25K positive and 25K negative 
- and the second difference is the labels we will use for the classification part are the actual labels
- but, for the language model, there are no labels, so we just use a bunch of zero's; that just makes it easier so we can use a consistent dataframe, or csv format
- now, the language model, we can create our own validation set, so you have probably come across by now `sklearn.model_selection.train_test_split(np.concatenate([trn_texts, val_texts]), test_size = 0.1)` which is a simple little function which grabs a dataset and randomly splits it into a training set and validation set, according to whatever proportion is specified by `test_size=0.1` (10%)
- in this case, I concatenate my classification training and validation together, so it is 100K together, split it by 10%, now I've got 90K training, 10K validation for my language model, so go ahead and save that
```python
trn_texts,val_texts = sklearn.model_selection.train_test_split(
    np.concatenate([trn_texts,val_texts]), test_size=0.1)
```
- that's my basic setup, get my data in a standard format for my language model and my classifier

### Language Model Tokens `00:28:00`
- the next thing we do is tokenization
- tokenization means, for a document, we've got a big long string, and we want to turn it into a **list of tokens**, which are *kind of*, a list of words, but not quite.  For example "don't", we want it to be:  "do n't" and a period / "<full stop>" to also be a token, and so forth
- so, tokenization is something we passed off to a terrific library called [spacy](https://spacy.io), partly terrific because an Australian wrote it, [Matthew Honnibal](https://twitter.com/honnibal), and partly terrific because it is good at what it does
- we put some work on top of spacy, but the vast majority of work done has been by spacy
- before we pass it to spacy, JH has written this simple **fixup** function which... each time JH opens a dataset, and has looked at *many*, everyone had different weird things that needed to be replaced.
- Here are all the ones JH has come up with so far:
```python
re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))
```
- hopefully, this will help you out as well
- `html_escape` all the entities
- and then there's a bunch more things that get replaced
- have a look at the result of the text you are using and make sure there are not more weird tokens in there, it's amazing how many weird things people do to text
- `get_texts` function

```python
def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)
```

#### 
- this function called `get_all` which will call `get_text` which calls `fixup`
- let's look thru this because there are some interesting things to point out
```python
def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels
```
- I'm going to use pandas to open our `train.csv` from the language model path (`LM_PATH`)
- I am passing an extra parameter you may not have seen before `chunksize` 
- Python and Pandas can both be pretty inefficient when it comes to storing and using text data
- You will see that very few people working with NLP are using large corpuses, and I think part of the reason is that traditional tools have made it difficult; you run out of memory all the time
- So, this process I am showing you today I have used on corpuses of over a billion words, successfully using this exact code
```python
df_trn = pd.read_csv(LM_PATH/'train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv(LM_PATH/'test.csv', header=None, chunksize=chunksize)
```
- and so, one of the simple tricks is to use this thing called `chunksize` with pandas
- what this means is pandas does not return a dataframe, but it returns an iterator which we can iterate thru chunks of a dataframe
- that's why JH doesn't say `tok_trn` = get_texts; instead JH calls `get_all`, which loops through the dataframe; but what it is actually doing is **looping through chunks of the dataframe**.
- each of those chunks is a dataframe representing subsets of the data
- Rachel:  when I'm working with NLP data, many times I come across data with foreign text or characters.  Is it better to discard them or keep them?
- JH:  No, definitely keep them.  And this whole process is unicode and JH has used it on Chinese text.  And it is designed to work on pretty much anything;  
- In general, most of the time it's not a good idea to **remove anything**.  Old fashioned NLP approaches tend to do things like lemmatization, and all these normalization steps to get rid of things, like lower case everything, blah, blah, blah etc.  
- But, **that's throwing away information** which you don't know ahead of time whether it's useful or not. 
- So, **don't throw away information**
- `00:32:20` so, we go through each chunck, `get_all`, each of which is a dataframe;  and we call `get_texts`.  `get_texts` is going to grab labels, make them into ints, it's going to then grab texts, and I'll point out a couple of things:
  1.  the first is that before we include the text, we have this `\n{BOS}` beginning-of-stream token, which you might remember we used way back earlier.  There's nothing special about these strings of letter.  I just figure they don't appear in normal text very often.  So, every text will start with `xbos`.  Why is that?  Because often it is really useful for your model to know when a new text is starting.  For example, if it is a language model, we are going to concatenate all the text together and so it would be very helpful for it to know when one article finishes and a new one started, so I should forget some of that context now. 
  - ? Ditto, Devo is quite often text has multiple fields, like a title, abstract and then the main document.  And so, by the same token, I've got this thing here (`get_texts`) which actually lets us have multiple fields in our csv.  
  - So this process is designed to be very flexible, and again, at the start of each one we put a special field starts here token, followed by the number of the field that's starting here, for as many fields as we have
  - then we apply fixup to it, and then most importantly, we tokenize it, and we tokenize it by doing a process or multiprocessor, or multi-processing, I should say.  
  - And, so, tokenizing tends to be pretty slow, but we've all got multiple cores on our machines now, and some of the better machines on AWS

