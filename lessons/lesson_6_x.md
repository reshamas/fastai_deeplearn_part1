# Lesson 6

[Livestream: Lesson 6](https://www.youtube.com/watch?v=sHcLkfRrgoQ&feature=youtu.be)

[Wiki: Lesson 6](http://forums.fast.ai/t/wiki-lesson-6/8629)

Notebooks:  
* [lesson5-movielens.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb)
* [lesson6-rnn.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson6-rnn.ipynb)
* [lesson3-rossman.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)

## Blogs to Review

* [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/index.html) by Sebastian Ruder (researcher, not USF student)  
  - this blog covers SGD, ADAM, weight decays :red_circle: (read it!)

* [Deep Learning #4: Why You Need to Start Using Embedding Layers](https://towardsdatascience.com/deep-learning-4-embedding-layers-f9a02d55ac12)


## Papers to Review
* [Entity Embeddings of Categorical Variables](https://www.slideshare.net/sermakarevich/entity-embeddings-of-categorical-variables)

## Summary of Course so Far
- our penultimate lesson


### Dimensions
- we can compress high dimensional spaces to a few dimensions, using PCA (Principal Component Analysis)
- PCA is a linear technique
- Rachel's computational linear algebra covers PCA
- PCA similar to SVD (singular value decomposition)
- find 3 linear combinations of the 50 dimensions which capture as much of the variation as possible, but different from each other
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
movie_pca = pca.fit(movie_emb.T).components_
```

### MAPE (Mean Average Percent Error)
- can give folks at work random forest with embeddings without using neural networks
- you can train a neural net with embeddings; everyone else in organization can chuck that into GBM or random forests or KNN
- can give power of neural nets to everyone in organization without everyone having to do fastai table
- embedding can be in SQL table
- GBM and random forests learn a lot quicker than neural nets do
- visualizing embeddings can be interesting
  - first, see things you expect to see
  - then, try seeing things that were not expected (some clusterings)
- Q:  skipgrams, a type of embedding?
  - A:  skipgrams for NLP
  - say we have an unlabeled dataset, such as Google Books
  - the best way, in my opinion to turn an unlabeled (or unsupervised) problem into a labeled problem is to invent some labels
  - what they did in Word2vec is:  here's a sentence with 11 words in it: _ _ _ _ _ _ _ _ _ _ _ 
    - let's delete the middle word and replace it with a random word
    - example:  replace "cat" with "justice"
    - sentence:  the cute little **CAT** sat on the fuzzy mat ---> **assign label = 1**
    - sentence:  the cute little **JUSTICE** sat on the fuzzy mat ---> **assign label = 0**
    - ! now we have something we can build a machine learning model on
    - quick, shallow learning, end up with embeddings with linear characteristics
    
## NLP
- for something more predictive, use neural net
- we need to move past Word2Vec and GLoVe, these linear based methods; these embeddings are way less predictive than with embeddings learned with deep models
- nowadays, **unsupervised learning** is really **fake task labeled learning**
- we need something where the type of relationships it's going to learn are the types we care about.

## Fake Tasks
- in computer vision, let's take an image and use an unusal data augmentation, such as recolor them too much, and ask neural net to predict augmented and non-augmented image
- use the best fake task you can
- a bad "fake task" is an **auto-encoder**; reconstruct my input using neural net with some activations deleted; most uncreative task, but it works surprisingly well
- we may cover this unsupervised learning in Part 2, if there is interest

https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb



