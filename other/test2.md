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
