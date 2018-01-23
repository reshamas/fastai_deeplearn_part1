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
      
### How Kaggle Compute Their Validation Score
- splits the test set into 2 pieces:  Public, Private
- they don't tell you which is which
- you submit your predictions to Kaggle
- Kaggle selects a random 30% to tell you your Leaderboard score
- at the end of the competition, that gets thrown away
- then they use the other 70% to calculate your "real score"
- making sure that you're not using the feedback from the Leaderboard to figure out some set of hyperparameters that do well but don't generalize
- this is why it is good practice to use Kaggle; at the end of a competition, you may drop 100 places in a competition
- good to practice on Kaggle than at a company where there are millions of dollars on the line

### Q:  case of not using a random sample for validation
- Q:  When might I not be able to use a random set for validation?
- cases:  in the case of temporal data, unbalanced data
- Tyler:  we expect things close by in time to be related close to them.  If we destroy the order, ...
- JH:  important to remember, when you buid a model, think that we are going to use the model at a time in the future
- when you build a model, you always have a systematic error, that the model will be used at a later time, at which time the world will be different than it is now; there is a lag from when time model is built to time when it is used; even when building the model, data is much older; a lot of the time, _that matters_
- if we're predicting who will buy toilet paper in NJ, and it takes us 2 weeks to put model in production, and we used data based on past 2 years, then by that time, things may look very different
- particularly, our validation set (if we randomly sampled from a 4-yr period), then the vast majority of that data is over a year old, and it may be that the toilet paper buying habits of folks in NJ may have dramatically shifted
  - maybe there is a terrible recession and they can't afford high quality paper
  - maybe paper making industry has gone thru the roof and they're buying more paper because it's cheaper
  - the world changes, if you use a random sample for your validation set, then you are actually checking:  how good are you at predicting things that are totally obsolete now?  how good are you at predicting things that happened 4 years ago?  That's actually not interesting.
- What you want to do is, practice every time there is a temporal piece

- 

  
  



---
## Topics
-  
