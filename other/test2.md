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
- use function `train_cats` to convert strings to pandas categories
- use function `set_categories` to re-order categories  

