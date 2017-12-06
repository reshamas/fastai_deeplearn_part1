# Download Dataset using `curl`

## Getting Data

Sample data:  https://www.kaggle.com/c/bluebook-for-bulldozers  
- in this example, we're using Firefox as a browser (you can use another browser)
- go to the website where the data is
- go to Developer section
  - method 1:  Javascript console, Developer 
  - method 2:  <kbd> ctrl + shift + i </kbd> to bring up web developer tool
- tab to Network
- go to data row
- right click, copy as "curl" (unix command that downloads data, like wget)
- might want to delete "2.0" in url since it causes problems
- <kbd> curl url_link -o bulldozers.zip -o </kbd> means output, then give suitable file name

## Setting up the Data Directory
- <kbd> mkdir bulldozers </kbd>
- <kbd> mv bulldozers.zip bulldozers/ </kbd>
- <kbd> sudo apt install unzip or brew install unzip </kbd>
- <kbd> cd bulldozers </kbd> 
- <kbd> unzip bulldozers.zip </kbd>


