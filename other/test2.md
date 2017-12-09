# RF - Part 2

Length: 01:35  
Notebook:  [lesson1-rf.ipynb](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson1-rf.ipynb)  

---

## Create a symlink
```bash
ln -s ../../fastai ./
```  
where `./` is the current directory
  
  
Evaluation Metric is:  root mean squared log error  
sum{ [(ln(act) - ln(pred)]^2 }  

## Data Process  
- we need all of our columns to be numbers
- use function `add_datepart` to replace a date variable with all of its date parts
- use function `train_cats` to convert strings to pandas categories (Notice: data type is not `string`, but `category`)
- use function `set_categories` to re-order categories  
- use function `proc_df` to replace categories with their numeric codes, handle missing continuous values, and split the dependent variable into a separate variable.
  >df, y, nas = proc_df(df_raw, 'SalePrice')
  - for continuous variables, missing values were replaced with the median

## R^2
- if you get an R^2 that is negative, it means your model is worse than predicting the mean
- R^2 is not necessarily what you're trying to optimize
- R^2 how good is your model vs the naive mean model?

## Test and Validation Sets
- Creating a validation set is the most important thing you'll do in machine learning.
- Validation Set (first hold out set): use this to determine what hyperparameters to use
- Testing (second hold out set): I've done modeling, now I'll see how it works

## Random Forest code
- `n_estimators` = number of trees
- `n_jobs=-1` --> means create a separate job for each CPU that you have  
```python
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
```

## Random Forest Scores output
[training RMSE , validation RMSE, training R^2, validation R^2]
```bash
[0.1026724559118164, 0.33553753413792303, 0.9786895444439101, 0.79893791069374753]
```

## Bagging
- statistical technique to create a random forest
- Bag of Little Bootstraps, Michael Jordan
  - create 5 different models which are not correlated --> they offer different insights
  - build 1000 trees on 10 separate data points --> invididual trees will not be predictive, but combined they will
 
## Bootstrapping
- pick out n rows with replacement

## Out-of-Bag (OOB) Score
- very useful when we have only a small dataset
```python
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
```
[training RMSE , validation RMSE, training R^2, validation R^2, OOB R^2]
```bash
[0.10198464613020647, 0.2714485881623037, 0.9786192457999483, 0.86840992079038759, 0.84831537630038534]
```

## Grid Search
- pass in list of hyperparameters we want to tune and values we want to try

## Subsampling
- The basic idea is this: rather than limit the total amount of data that our model can access, let's instead limit it to a different random subset per tree. That way, given enough trees, the model can still see all the data, but for each individual tree it'll be just as fast as if we had cut down our dataset as before.
- no dataset is too big for this technique (ex: 120 million rows for grocery store data of Kaggle competition)
- need to set `oob_score = False` if using subsample approach of `set_rf_samples(20000)`
- to turn it off, do `reset_rf_samples()`
```python
set_rf_samples(20000)
```

## Important Takeaway / Tip
- very few people in industry or academia do this
- most people run all of their models on all of their data all of the time using their best parameters
- do most of your models on a large enough sample size so your accuracy is reasonable, that takes a small number of seconds to train

## Tree Building Parameters
- `min_samples_leaf=1` this is the default
- `min_samples_leaf=3` says stop training the tree further when your leaf node has 3 or less samples in; the numbers 1, 3, 5, 10, and 25 work well
- `max_features=None` this is the default; then max_features=n_features (default is to use all the features)
- `max_features=0.5` the less correlated your trees are with each other, the better; randomly choose half the features
- `max_features` in practice, good values range from 0.5 to log2 or sqrt




