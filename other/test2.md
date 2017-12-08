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

## Random Forest Scores output
[xxx, xxx, training R^2, validation R^2]
```bash
[0.1026724559118164, 0.33553753413792303, 0.9786895444439101, 0.79893791069374753]
```
