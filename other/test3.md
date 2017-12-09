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

## Example of Large dataset
### Grocery Store Competition
Brick-and-mortar grocery stores are always in a delicate dance with purchasing and sales forecasting. Predict a little over, and grocers are stuck with overstocked, perishable goods. Guess a little under, and popular items quickly sell out, leaving money on the table and customers fuming.

The problem becomes more complex as retailers add new locations with unique needs, new products, ever transitioning seasonal tastes, and unpredictable product marketing. Corporación Favorita, a large Ecuadorian-based grocery retailer, knows this all too well. They operate hundreds of supermarkets, with over 200,000 different products on their shelves.

Corporación Favorita has challenged the Kaggle community to build a model that more accurately forecasts product sales. They currently rely on subjective forecasting methods with very little data to back them up and very little automation to execute plans. They’re excited to see how machine learning could better ensure they please customers by having just enough of the right products at the right time.

#### Current Ecuador grocery store competition
* [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)
- this is current, so we won't work on it as a group
- predict items on shelf based on 
- oil prices, stores, locations
- ability to explain the problem is very important
- key:  what are **independent** variables- how many units of each kind of product are sold on each store on each day during a 2 week period?
- **dependent**, info we have to predict it:  how many units of each product of each store on each day was sold; 
- what **metadata** is there (oil price) --> our **relational dataset**
- 






