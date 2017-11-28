# Jupyter Notebook

In [Kaggle 2017 data science survey](https://www.kaggle.com/surveys/2017) of 16K data scientists, Jupyter Notebook came up as 3rd most important self-reported tool for data science.  

## Notebook Features
* can add text, images, code - all in one place
* can document what we're doing as we go along and code
* can put pictures, videos, html tables, interactive widgets
* great experimentation environment

## Help
<kbd> h </kbd> shows the list of shortcuts

## Notebook Commands / Shortcuts
* <kbd> Shift + Enter </kbd> to run cell  
* <kbd> Shift + Tab </kbd>  First time pressing:  tells you what parameters to pass 
* <kbd> Shift + Tab </kbd> Press 3 times:  gives additional info about method

### Select multiple cells 
<kbd> ESC </kbd>    <kbd> Shift+ :arrow_up: </kbd>  extend select cells above  
<kbd> ESC </kbd>   <kbd> Shift+ :arrow_down: </kbd>  extend select cells below  


## Notebook Source Code Access
* <kbd> ? </kbd>  look at documentation for code
  * Example: <kbd> ?ImageClassifierData.from_paths </kbd>
* <kbd> ?? </kbd>  look at source code
  * Example:  <kbd> ??ImageClassifierData.from_paths </kbd>
* type: <kbd> function name </kbd>, then <kbd>Shift + Enter </kbd> to find out where a particular function or class comes from 
  * Example of Input:  <kbd> ImageClassifierData </kbd> <kbd>Shift + Enter </kbd>
  * Example of Output: `fastai.dataset.ImageClassifierData`
* Within function, <kbd>Shift + Tab </kbd> shows parameters that the function can take, also shows default parameter options
* `object`, then <kbd> Tab </kbd> shows all the options for that object or function

## Convert your notebooks to .md 
<kbd> jupyter nbconvert --to <output format> <input notebook>  </kbd>


---
## Resources

* [How to Change Your Jupyter Notebook Theme](https://jcharistech.wordpress.com/2017/05/18/how-to-change-your-jupyter-notebook-theme/)
