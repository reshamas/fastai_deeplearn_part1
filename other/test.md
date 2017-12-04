
# try 1 - random forests
https://www.kaggle.com/c/bluebook-for-bulldozers

- ML should help us understand a dataset, not just make predictions about it.

Firefox, to website, then Javascript console, Developer
- ctrl + shift + i to bring up web developer tool
- tab to Network
- go to data row
- right click, copy as Curl (unix command that downloads data, like `wget`)
- might want to delete "2.0" in url since it causes problems
- `curl url_link -o bulldozers.zip` `-o` means output, then give suitable file name
- `mkdir bulldozers`
- `mv bulldozers.zip bulldozers/`
- `sudo apt install unzip` or `brew install unzip`
- `unzip bulldozers.zip`

Python 3.6 format string:  
```python
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, 
                     parse_dates=["saledate"])
``` 
- `f'{PATH}Train.csv'`  the `f` tells it to interpolate the "{PATH}"
- `low_memory=False` make it read more of the file to decide what the types are  

### Example
`name = 'Jeremy'`  
`age = 43`  
`f'Hello {name.upper()}, you are {age}'`  
output:  
>Hello, Jeremy, you are 43  

### Random Forest
- universal machine learning technique
- way of predicting something of any kind (dog/cat, price)
- can predict a categorical or continuous variable
- columns can be of any kind (pixel data, zip codes, revenues)
- in general, it doesn't overfit
- easy to stop it from overfitting
- don't need a separate validation set
- has few, if any statistical assumptions
  - doesn't assume data is normally distributed
  - doesn't assume relationships are linear
  - don't need to specify interactions
- requires few pieces of feature engineering (don't have to take log of data)
- it's a great place to start
- if your random forest doesn't work, it's a sign there is something wrong with the data





