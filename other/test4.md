# test 4

Length: 01:40  

Notebook:  [lesson2-rf_interpretation.ipynb](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson2-rf_interpretation.ipynb)  

---

## Topics
- R^2 accuracy
- How to make validation sets
- Test vs. Validation Set
- Diving into RandomForests
- Examination of One tree
- What is 'bagging’
- What is OOB Out-of-Box score
- RF Hyperparameter 1: Trees
- RF Hyperparameter 2: max Samples per leaf
- RF Hyperparameter 3: max features

## Repository / Notebook Workflow
- make a copy of the notebook
- name it with `tmp` prefix; this will then be ignored by `.gitignore`

## Hyperparameter `set_rf_samples()`  
- pick up a subset of rows
- summarize relationship between hyperparameters and its effects on overfitting, collinearity
- reference:  https://github.com/fastai/fastai/blob/master/courses/ml1/lesson1-rf.ipynb
- `set_rf_samples(20000)` determines how many rows of data in each tree
  - Step 1: we have a big dataset, grab a subset of data and build a tree
  - we either bootstramp a sample (sample with replacement) or subset a small number of rows
  - Q:  assuming the tree remains balanced as we grow it, how many layers deep would we want to go?  
    - A: log_2(20,000)  (depth of tree doesn't really vary based on sample size)
  - Q:  how many leaf nodes would there be?
    - A: 20,000  (because every leaf node would have a sample in it)
  - when you decrease the sample size, it means that there are less final decisions that can be made; tree will be less rich; it also is making less binary choices to get to those decision
  - setting `set_rf_samples` lower means you overfit less, but you'll have a less accurate tree model
  - each individual tree (aka "estimator") is as accurate as possible on the training set
  - across the estimators, the correlation between them is as low as possible, so when you average them out together, you end up with something that generalizes
  - by decreasing the `set_rf_samples()` number, we are actually decreasing the power of the estimator and increasing the correlation
  - it may result in a better or worse validation set result; this is the compromise you have to figure out when you do ML models
 - `oob=True` whatever your subsample is, take all the remaining rows, and put them into a dataset and calculate the error on those (it doesn't impact the training set); it's a quasi-validation set

## Information Gain
- "Information" used to describe the amount of additional info we gain from splitting
- how much better did the model get by adding another split point?

## Hyperparameter `min_samples_leaf`  
- Q:  if I change min_samples_leaf from 1 to 2, what would be my new **depth**?
  - A:  log_2(20,000) - 1
- Q:  how many leaf nodes would there be in that case?
  - A:  10000
- we have less depth, less decisions to make, and we have a smaller number of leaf nodes
  - we would expect each estimator to be less predictive, but also less correlated and result in less overfitting
- could speed up training with one less level; could generalize better
- TRY these options
  - 1, 3, 5, 10, 25, 100

## Hyperparameter `max_features`  
- `max_features=0.5` at each point in the tree, we pick a different half of the features 
- we do this because we want the trees to be as rich as possible
- picking a random subset of features at every decision point
- overall effect is that each individual tree will be less accurate, but the trees will be more varied
  - imagine if you had one feature that was super-predictive, so predictive that every single sub-sample split on the same feature
  - trees would have same intial split
  - some trees would create other splits, show interactions
  - gives more variation, creates more generalized trees
- TRY these options
  - `None`
  - `0.5`
  - `sqrt`

## Things that don't impact our training
- `n_jobs=-1` how many CPUs to run on
  - making more than 8 may have diminishing returns
  - -1 --> all cores
  - 1  --> default
- `oob=True` if you don't say "True", it won't print it out

## Other Parameters
- there are more hyperparameters
- the one's highlighted here are the ones that Jeremy has found useful
- 


## Random Forest Model Interpretation
