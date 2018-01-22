# Lesson 5

Length: 01:40  

Notebook:  [lesson2-rf_interpretation.ipynb](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson2-rf_interpretation.ipynb)  

---
## Review
- What's the difference between Machine Learning and "any other kind of [analysis] work"?  In ML, we care about the **generalization error** (in other analysis, we care about how well we map our observations to outcome)
- the most common way to check for **generalization** is to randomly pull some data rows into a **test set** and then check the accuracy of the **training set** with the **test set**
- the problem is: what if it doesn't generalize?  could change hyperparameters, data augmentation, etc.  Keep doing this until many attempts, it will generalize.  But after trying 50 different things, could get a good result accidentally
- what we generally do is get a second **test set** and call it a **validation set**
- a trick for **random forests** is we don't need a validation set; instead, we use the **oob error/score (out-of-bag)**
  - every time we train a tree in RF, there are a bunch of observations that are held out anyway (to get rid of some of the randomness).  
  - **oob score** gives us something pretty similar to **validation score**, though, on average, it is a little less good
  - samples from oob are bootstrap samples
  - ** with validation set, we can use the whole forest to make the prediction
  - ** but, here we cannot use the whole forest; every row is going to use a subset of the trees to make its predictions; with less trees, we get a less accurate predction 
  - ** think about it over the week
  - Why have a validation set at all when using random forests?  If it's a randomly chosen validation dataset, it is not strictly necessary; 
    - you've got 4 levels of items we're got to test
      1.  oob - when that's done working well, go to next one
      2.  validation set
      3.  test
      
### Kaggle splits the test set into 2 pieces:  Public, Private
- they don't tell you which is which
- 
  
  



---
## Topics
-  
