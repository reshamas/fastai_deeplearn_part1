# Lesson 5

[Video: Lesson 5](https://www.youtube.com/watch?v=J99NV9Cr75I&feature=youtu.be)

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

#### Classifier
Q:  if recommendation system is 0 / 1?  
A:  
- need to use a classifier instead of regressor
- not in fastai library yet
- change activation function to be sigmoid, criterion (loss function) to cross entropy rather than RSME


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
  - all PyTorch modules are written in Python OO
  - a function which we'll be able to take a derivative of, stack on top of each other, etc.
- TensorFlow
  - has its own syntax
- `def forward()` 
- gradients are backwards calculations, we don't have to calculate that
- that's it, that's how we create a custom PyTorch layer
  ```python
  class DotProduct(nn.Module):
    def forward(self, u, m): return (u*m).sum(1)
  
  model=DotProduct()
  
  model(a,b)
  ```
```bash
  6
 70
 [torch.FloatTensor of size 2]
```

Q:  Can you use fastai library for *very large* datasets, for collaborative filtering
A:  
- yes, absolutely, it uses mini-batch stochastic gradient descent 
- this version will create a pandas dataframe, which needs to live in memory
- can easily get 5 to 12 Gig instances on AWS, if you have CSV file > 5-12 Gig, then you'll have to save it as big ?hols array or save as a Dask dataframe

#### [Initialization Of Deep Networks Case of Rectifiers](http://www.jefkine.com/deep/2016/08/08/initialization-of-deep-networks-case-of-rectifiers/)  
Mathematics Behind Neural Network Weights Initialization - Part Three: In this third of a three part series of posts, we will attempt to go through the weight initialization algorithms as developed by various researchers taking into account influences derived from the evolution of neural network architecture and the activation function in particular.
   
#### Back to PyTorch Model
```python
class EmbeddingDot(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.u.weight.data.uniform_(0,0.05)
        self.m.weight.data.uniform_(0,0.05)
        
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        u,m = self.u(users),self.m(movies)
        return (u*m).sum(1)
```

This PyToch notation here means:  

```python
        self.u.weight.data.uniform_(0,0.05)
        self.m.weight.data.uniform_(0,0.05)
```

- `self.u` is an instance of the embedding class
- `self.u.weight` is an attribute of `self.u` which contains the actual embedding matrix
- the actual embedding matrix is not a tensor, it is a variable
- a variable is exactly the same as a tensor, that is, it supports the exact same operations, but it also does **automatic differentiation**
- to pull the tensor out of a variable, you get its data attribute:  `self.u.weight.data` --> so, this is now the tensor of the weight matrix of the of the `self.` embedding
- something that's really handy to know is that all of the tensor functions in PyTorch, you can stick an underscore `_` at the end, and that means does it **in-place** 
- `self.u.weight.data.uniform_(0,0.05)` create a uniform random number of an appropriate size for this tensor and don't return it but actually fill in that matrix in-place
- here is the **Non in-place version:**  `self.u.weight.data.uniform = self.u.weight.data.uniform_(0,0.05)`

**Note:**  PyToch can do mini-batch at a time with pretty much everything that we can get really easy speed up.  We don't have to write any loops on our own.  If you ever do loop through your mini-batch manually, you don't get GPU acceleration.   (*don't use for-loops here*)

Type `??fit` to find out what this function does:  `fit(model, data, 3, opt, F.mse_loss)`  

class break at 58:33  

This is where the code is:  [model.py](https://github.com/fastai/fastai/blob/master/fastai/model.py)  
```python
    for epoch in tnrange(epochs, desc='Epoch'):
        stepper.reset(True)
        t = tqdm(iter(data.trn_dl), leave=False, total=len(data.trn_dl))
        for (*x,y) in t:
            batch_num += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = stepper.step(V(x),V(y))
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)
            t.set_postfix(loss=debias_loss)
            stop=False
            for cb in callbacks: stop = stop or cb.on_batch_end(debias_loss)
            if stop: return
```
Q: what if we want to squish the ratings between 0 and 5?  
A: put rating through sigmoid function (which has range (0, 1) and multiply by 5; or multiply by 4 and add 1)  
Note:  it is good practice to do this, set range between (min_rating, max_rating).  

```python
import torch.nn.functional as F
```
`.cuda()` is we tell it manually to use the GPU
```python
wd=1e-4
model = EmbeddingNet(n_users, n_movies).cuda()
opt = optim.Adam(model.parameters(), 1e-2, weight_decay=wd)
```
### Looking at Code
[column_data.py](https://github.com/fastai/fastai/blob/master/fastai/column_data.py)

## Neural Net version of Collaborative Filtering

## Derivatives
- Backpropagation is the same as "apply the chain rule to all the layers"
- ReLU ---> same as saying replace the negatives with zero!

SGD w/momentum vs ADAM  
student, Anand Saha, is working on prototype ADAMw, confirming he is getting faster performance and better accuracy.  

### Adaptive Learning Rate

### Weight Decay
- helpful regularizer

## Homework
- practice the thing you're least familiar with

Watch end of video a few times to understand, at 01:30  


