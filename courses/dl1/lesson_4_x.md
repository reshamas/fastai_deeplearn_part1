# Lesson 4:  Structured Neural Net Intro, Language RNN Intro, Collaborative Filtering Intro
(20-Nov-2017, live)  
 
[Wiki: Lesson 4](http://forums.fast.ai/t/wiki-lesson-4/8112)

Notebook:  
* [lesson3-rossman](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)
* [lang_model-arxiv.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lang_model-arxiv.ipynb)
---

## Blogs to Review
* [Improving the way we work with learning rate](https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b) Vitaly Bushaev
* [Cyclical Learning Rate Technique](http://teleported.in/posts/cyclic-learning-rate/) Anand Saha
* [Exploring Stochastic Gradient Descent with Restarts (SGDR)](https://medium.com/38th-street-studios/exploring-stochastic-gradient-descent-with-restarts-sgdr-fa206c38a74e) Mark Hoffman (nice intro level)
* [Transfer Learning using differential learning rates](https://towardsdatascience.com/transfer-learning-using-differential-learning-rates-638455797f00) Manikanta Yadunanda
* [Getting Computers To See Better Than Humans](https://medium.com/@ArjunRajkumar/getting-computers-to-see-better-than-humans-346d96634f73) Arjun Rajkumar (technology plus its implications)
* [**Rachel Thomas**:  How (and why) to create a good validation set](http://www.fast.ai/2017/11/13/validation-sets/)

Just jump in and write a technical post.

## Topics to Cover in this Lesson
* Structured neural net info
  - building models on database tables
* Language RNN Intro
  - NLP
* Collaborative Filtering Info
  - recommendation systems

### Focus on this lesson 
- here's the software to do it
- in the next few lessons, we'll get into more details behind scences
- also, we'll look at the details of computer vision

---
# Dropout
- looking at Kaggle [dog breed competition](https://www.kaggle.com/c/dog-breed-identification)
- 
type learner object, and you can see the layers:  
`learn`  
`Sequential (`
- `(0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)` we'll do this later
- `(1): Droput (p = 0.5)` 
- `(2): Linear (1024 -> 512)`  linear layer, matrix multiplier.  1024 rows and 512 columns; take in 1024 activations and spit out 512 activations  
- `(3): ReLU ()`  replace negatives with 0
- `(4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)` we'll do this later
- `(5): Droput (p = 0.5)`
- `(6): Linear (512 -> 120)` takes 512 activations, puts thru new matrix multiplier (512 x 120), outputs 120
- `(7): LogSoftmax () `  

`)`

<img src="../images/softmax.png" align="left" height="280" width="700" >   <br> <br>

### Dropout of p = 0.5
- p = probability of deleting the cell
- notice that when half the activations are randomly deleted, it doesn't really change the output 
- the act of randomly throwing away half the activations has an interesting result
- each minibatch, we throw away a different random part of activations in that layer
- throwing the activation randomly away forces it to **not overfit**
- it forces it to work even if half the activations are randomly thrown away
- has been critical in making modern deep learning work because it solved the problem of **generalization** for us
- dropout is about 3-4 years old now
- Before **dropout**, if we tried to train a model with lots of parameters and you were overfitting, and you had tried data augmentation or more data, you were stuck; Geoffrey Hinton came up with dropout idea, loosely inspired by how brain works
- p = 0.01 not so effective
- p = 0.99 not so effective, would kill your accuracy; high p vals generalize well, but lower training accuracy

Why is that early in training, my validation losses are better than my training losses?  
- when we look at validation set, we turn off **dropout**  

Do you have to do anything to accommodate that we are throwing away some activations via dropout?  
- PyTorch behind the scenes, if p = 0.5, throws away the activations, but it doubles the activations that are there, so the average activation does not change  

Can run model 2 ways, one with `ps=0.0` and another with `ps=0.5`.  Compare models, one with dropout and one without, and see results.  

You always need 1 linear layer: to connect input with the output.  
`xtra_fc=[ ]` can pass a list of many of the fully connected layers to be  
`xtra_fc=[ ]` passing in an empty list means there will be only 1 fully connected layer  
If we have dropout = 0 (`ps=0`) and no extra fully connected layers (`xtra_fc=[]`):  
```python
learn = ConvLearner.pretrained(arch, data, ps=0., precompute=True, xtra_fc=[] )
learn.fit(1e-2, 3) 
learn
```  
output is this.  It is the minimal possible top model.    
```
Sequential (
(0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)
(1): Linear (1024 -> 512)
(2): LogSoftmax ()
```  
Note:  with bigger models, (resnet34 has few parameters) like resnet50, you will need more dropout.  
Q:  Is there a way to see if the data is being overfit?  
A:  Yes, when the training error loss is much lower than the validation error loss.  

---
# Structured Data Problem
Notebook:  [lesson3-rossman](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)

### Categorical and Continuous
Which vars are categorical and which are continuous is a modeling decision you get to make.  
- if categorical in data --> have to call it categorical
- **cardinality** is how many levels are in a category, (ex: cardinality for days of week is 7)
- if it starts off as continuous, such as day of week, you get to decide
- if continuous in data --> you get to pick
  - things like Year, it often works better to make it categorical
- continous vars: are actual floating point numbers; hard to make these categorical because they have many, many levels
  
```python
cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
    'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
    'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
    'SchoolHoliday_fw', 'SchoolHoliday_bw']

contin_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
   'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 
   'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
   'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']

n = len(joined); n
```
Jeremy so far has not binned continuous variables.  Though, a paper came out this week to the contrary.  Requires further research.  

PyTorch expects continuous to be type float32.  
```python
for v in cat_vars: joined[v] = joined[v].astype('category').cat.as_ordered()

apply_cats(joined_test, joined)

for v in contin_vars:
    joined[v] = joined[v].astype('float32')
    joined_test[v] = joined_test[v].astype('float32')
```

#### process dataframe
- `proc_df` this is part of fastai library;  "process dataframe"  
- pull Sales out of dataframe and sets it as y variable
- `do_scale=True` Scaling - neural nets really like input data to be ~N(0, 1)
- `nas` handles missing values; continuous - replace missing with median
- 
```python
df, y, nas, mapper = proc_df(joined_samp, 'Sales', do_scale=True)
yl = np.log(y)
```

## Embedding
- would build an embedding matrix for each categorical feature of the data
- it is called `emb_szs`
- setting up initial embeddings
  - randomized
  - pretrained
- idea of embedding is actually what's called a **distributed representation**, fundamental concept of neural network

## NLP
Language Modeling - build a model where given a few words of a sentence, predict what the next few words will be.

Notebook:  https://github.com/fastai/fastai/blob/master/courses/dl1/lang_model-arxiv.ipynb

Arxiv:  https://arxiv.org  
most popular pre-print server; has lots of academic papers   

`import torchtext`  
Torchtext is PyTorch's NLP.  
`bptt` back prop through time  

## Next Week
Collaborative Filtering:  https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb









