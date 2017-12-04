
# try 1
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


