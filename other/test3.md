# test 3

Length: 01:24  

Notebook:  [lesson2-rf_interpretation.ipynb](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson2-rf_interpretation.ipynb)  

---

## Last Lesson
- random forests
- random forests parameter tuning to make them better
- we used Jupyter Notebook; can use Anaconda, AWS, Crestle, Paperspace

## This Lesson
- start with `git pull` for updates

## Interpreting Model
- understand your data better using machine learning
- not true that random forests are a "black box" 
- random forests allow us to understand our data, deeper and more quickly
- we can also look at larger datasets

Q:  When do I know to use random forests?  
A:  It's always worth trying.  

Q:  In what situations should I try other things as well?  
A:  For **unstructured data**, such as audio, NLP or images, will want to use deep learning  
A:  Outside of that, will also want to use collaborative filtering modeling  (neither Random Forests or Deep Learning)  

Last week, saved dataframe to "feather" format; basically that is in same format as it is in RAM, but it is ridiculously fast to read and write stuff from feather format.  

## Data Pre-Processing
loading in last lesson:  
```python
PATH = "data/bulldozers/"
df_raw = pd.read_feather('tmp/raw')
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
```
### `proc_df` (pre-processing data)
- find the numeric columns which have missing values and creates an additional boolean, replaces missing with medians
- turns categorical objects into integer codes
- test set may have missing values.  random forest will give an error if a column is missing
- median of missing values may be different in test set than in training set
- `nas`:  Jeremy changed `proc_df` so it returns "NAs"; `nas` is a dictionary where keys are the names of the columns with missing values and the values of the dictionary are medians
- you can pass `nas` as an additional argument to `proc_df` and it will make sure it uses those particular columns and gives those medians
- Jeremy will update the rest of course notebooks to assign third output of "nas"
- 







