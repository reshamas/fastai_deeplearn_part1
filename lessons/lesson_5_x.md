# Lesson 5

[Livestream: Lesson 5](https://www.youtube.com/watch?v=J99NV9Cr75I&feature=youtu.be)

[Wiki: Lesson 5](http://forums.fast.ai/t/wiki-lesson-5/8408)  

Notebooks:  [lesson5-movielens.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb)

## Blogs to Review

* [Structured Deep Learning](https://towardsdatascience.com/structured-deep-learning-b8ca4138b848) Kerem Turgutlu (Masters' student at USF)
  - experiment with different Kaggle datasets
  - experiment with dropout, etc
  - unexplored territory
* [Fun with small image data-sets (Part 2)](https://medium.com/@nikhil.b.k_13958/fun-with-small-image-data-sets-part-2-54d683ca8c96) Nikhil B
* [How do We Train Neural Networks?](https://towardsdatascience.com/how-do-we-train-neural-networks-edd985562b73) Vitaly Bushev
  - great technical communication

## Summary of Course so Far
### First Half of this Course
- getting through applications for use
- here's the code you have to write
- high level description of what code is doing 

### Second Half of this Course
- now, we're going in reverse; digging into detail
- dig into source code of fastai library and replicate it
- no more best practices to show us

### Today's Lesson
- create effective collaborative filtering model from scratch
- will use PyTorch as an automatic differentiating tool (won't use neural net features)
- will try not to use fastai library

## Collaborative Filtering
- Movie Lens Dataset:  [lesson5-movielens.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb)  
- Excel file is here:  https://github.com/fastai/fastai/blob/master/courses/dl1/excel/collab_filter.xlsx

We will use 3 columns of data:   
- userId (categorica)
- movieId (categorical)
- rating (continous, dependent var)

Can use this data later:  
- movie title
- genre

### Matrix Factorization
- we're not going to start with neural network, because there is a simple way to work on these problems
- doing "shallow learning" in excel spreadsheet
- using top 15 raters and top 15 movies rated
- for each cell of userID by movieID, there is a corresponding 1x5 row, and a 5x1 column (initially, the numbers are random)
- then, via Solver, we set the **Objective Function** to be the **Root MSE**
- if you've done linear algebra, this is matrix decomposition
- we can use gradient descent to solve
- Jeremy prefers to think of it from an intuitive point of view; 

### Working the Collaborating Filtering Example in Excel
- for userID 27, movieID 72
- for movieID 72, (Example: Lord of the Rings, Part 1) random weights might represent:  
  - how much is it scifi and fantasy
  - how recent a movie
  - how much special effects
  - how dialogue driven is it?
  - etc
- for userID 27
  - how much does the user like scifi and fantasy?
  - how much does the user like dialogue driven movies?
  - etc
- The problem is, we don't have that information.  So we're starting with random
  - these "factors", we call them **latent factors**, we've assumed we can think of movie ratings this way
  - then we use **gradient descent** to find some numbers that work
- this is collaborating filtering using **probabilistic matrix factorization**
- next, we will implement in Python, and run it on the whole dataset
- the weight vectors are **embedding matrices**
- how do we decide on the dimensionality of our embedding matrix?  
  - We have no idea and we have to try a few things and see what works
  - underlying concept is you need to pick an embedding dimensionality which is enough to reflect the true complexity of this causal system, but not so big that you have too many parameters, or it could take forever to run, or be overfit
- What does it mean if factor is negative?  Means it is not dialogue driven; for user, a dislike
- Q: if we have a new movie or new user, do we have to retrain the model?
  - A:  not a straighforward answer to that, time permitting, we'll come to that.  would need a new movie/user model
  
### Working the Collaborating Filtering Example in Python
- get validation set, use random indices
- `wd=2e-4` = weight decays; in ML, it is **L2 Regularization**
- `n_factors=50` how big an embedding matrix we are choosing
- `cf = CollabFilterDataset.from_csv(path, 'ratings.csv', 'userId', 'movieId', 'rating')` --> path, file, rows (users), columns (items), rating
- which other movies are similar to users who liked it
- which other people are similar who liked this movie
- `learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn=optim.Adam)` 
   - `n_factors` = size of embedding matrix
   - `val_idxs`  = what validation set indexes to use
   - batch size = 64 here
   - `opt_fn=optim.Adam` = what optimizer to use
   - fit the model:  `learn.fit(1e-2, 2, wds=wd, cycle_len=1, cycle_mult=2)`
- plots are in seaborn - plotting library that sits on top of matplotlib

### Dot Product Example in PyTorch
- dot product is the vector version
- can create a tensor in PyTorch version

```python
a = T([[1.,2],[3,4]])
b = T([[2.,2],[10,10]])
a,b
```
- PyTorch module:
  - uses Pythonic way of doing things
  - can use as a layer in a neural net
  - PyTorch uses Python OO (Object Oriented) way of doing things, defining classes, etc.
  - a function which we'll be able to take a derivative of, stack on top of each other, etc.
- TensorFlow
  - has its own syntax
- `def forward()` 
- `def backward()`


   
   
   


 




