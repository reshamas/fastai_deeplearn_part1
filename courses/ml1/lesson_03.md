# Lesson 3:  Preprocessing Data & Feature Importance

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

## Example of Large dataset
### Grocery Store Competition
Brick-and-mortar grocery stores are always in a delicate dance with purchasing and sales forecasting. Predict a little over, and grocers are stuck with overstocked, perishable goods. Guess a little under, and popular items quickly sell out, leaving money on the table and customers fuming.

The problem becomes more complex as retailers add new locations with unique needs, new products, ever transitioning seasonal tastes, and unpredictable product marketing. Corporación Favorita, a large Ecuadorian-based grocery retailer, knows this all too well. They operate hundreds of supermarkets, with over 200,000 different products on their shelves.

Corporación Favorita has challenged the Kaggle community to build a model that more accurately forecasts product sales. They currently rely on subjective forecasting methods with very little data to back them up and very little automation to execute plans. They’re excited to see how machine learning could better ensure they please customers by having just enough of the right products at the right time.

---

## Current Ecuador grocery store competition
* [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
- this is current, so we won't work on it as a group
- predict items on shelf based on 
- oil prices, stores, locations
- ability to explain the problem is very important
- key:  what are **independent** variables- how many units of each kind of product are sold on each store on each day during a 2 week period?
- **dependent**, info we have to predict it:  how many units of each product of each store on each day was sold; 
- what **metadata** is there (oil price) --> our **relational dataset**
- Stars schema  https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data
- Snowflake schema:  might have more info on the items
- Jeremy's notebook:  `tmp-grocery.ipynb`

## Notebook
- read in data
- limit memory = False --> use as much memory to figure out what kinds of data are here; you'll run out of memory here
- to limit the amount of space, create a dictionary for each column names
```python
types = {'id': 'int64',
'item_nbr': 'int32',
'store_nbr': 'int8',
'unit_sales': 'float32',
'onpromotion': 'object'}
```
- `df = pd.read_csv('somefile.csv', low_memory=False)`

### Process
- this dataset has 125 million rows
- use `head` function to look at small amount of data; determine the data types and set it up in a dictionary called "types" (see above)
  - or read in small dataset and let pandas figure it out for you
- can now read in data in less than 2 minutes
```python
%%time
df_all = pd.read_csv(f'{PATH}train.csv', parse_dates = ['date'], dtype=types, infer_datetime_format = True)
```
```python
set_rf_samples(1_000_000)
```
- there are 120 million records; we probably don't want to create a tree; would take a long time
- can start with 10K or 100K; Jeremy found 1 million is good size, runs in < 1 minute
- there is no relationship between how large a dataset is and how long it takes to build a random forest
- relationship is between number of estimators times sample size
- `n_jobs=8` number of cores it will use; Jeremy ran it on computer that had 60 cores, so make it smaller
- `n_jobs=-1` means use every single core
- Jeremy converted dataframe into array of float32; internally inside random forest code, they do that anyway
- by doing it once, saves time in the background to convert it to float
```python
%time m.fit(x, y)
%prun m.fit(x, y)
```
- `prun` runs **profiler**, tells you which line of code behind the scenes took the most time to run
- **profiling** is a software engineering tool
- cannot use `oob_score` when using `set_rf_samples`, because it will use 125 million - 1 million = 124 million too calculate oob_score, which will take forever
- wants validation set, which is the most recent date samples
- [training RMSE , validation RMSE, training R^2, validation R^2, OOB R^2]

### More Data
- on Kaggle, weather data is meta data about a date

#### Former German grocery store competition
- Kaggle competition:  [Rossman Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
- 2015 competition  

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied.

In their first Kaggle competition, Rossmann is challenging you to predict 6 weeks of daily sales for 1,115 stores located across Germany. Reliable sales forecasts enable store managers to create effective staff schedules that increase productivity and motivation. By helping Rossmann create a robust prediction model, you will help store managers stay focused on what’s most important to them: their customers and their teams! 

## Interpreting Machine Learning Models
- Notebook:  [lesson2-rf_interpretation.ipynb](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson2-rf_interpretation.ipynb)  

## Feature Importance

### Data Leakage
- there was an administrative burden to filling in the database
- there is information in the dataset that J was modeling with which the university wouldn't have had when the decision was made to approve the grant

### Collinearity
- generally speaking, removing redundant columns makes the R2 better
- 

## HW
- look further at model, data
