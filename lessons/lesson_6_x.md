# Lesson 6

[Livestream: Lesson 6](    )
https://www.youtube.com/watch?v=uXt9vwlAPjc


[Wiki: Lesson 6](   )  

Notebooks:  
* [lesson5-movielens.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb)
* [lesson6-rnn.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson6-rnn.ipynb)

## Blogs to Review

* [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/index.html) by Sebastian Ruder (researcher, not USF student)  
  - this blog covers SGD, ADAM, weight decays :red_circle: (read it!)

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
    
