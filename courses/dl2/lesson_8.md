# Lesson 8:  xxx
(19-Mar-2018, live)  
 
[Wiki: Part 2 / Lesson 8](http://forums.fast.ai/t/part-2-lesson-8-in-class/13556)

Notebook:  
*  [pascal.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/pascal.ipynb)

---

## Staff
* Intro by [David Uminsky](https://www.usfca.edu/faculty/david-uminsky), Director of Data Institute of USF 
* [Jeremy Howard](https://www.usfca.edu/data-institute/about-us/researchers), Distinguished Scholar in Deep Learning

### Notes
* 600 international fellows around the world
* Rachel & Jeremy will be in room 153, 10am to 6pm each day (not for mentoring, possible projects)

---
## Object Detection
* creating much richer convolutional structures
* what is a picture of and where it is in the picture


## Learning
* Jeremy trying to pick topics that will help us learn foundational topics (richer CNN)
* can't possibly cover hundreds of interesting things done with deep learning
* 

## Park 1 Takeaways
* we don't call this deep learning, but differential programming
* Part 1 was setting up a differential function, a loss function and pressing Go
* If you can configure a loss function that configures score, how good a task is, you're kind of done
* [playground.tensorflow.org](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.71280&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
  * play interactively where you can create and play with your functions manually

## Transfer Learning - definition
Transfer learning or inductive transfer is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.[1] For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. This area of research bears some relation to the long history of psychological literature on transfer of learning, although formal ties between the two fields are limited.
 


### Transfer Learning
* the most important thing to learn to do to use deep learning effectively
* it makes nearly everything easier, faster and more accurate
* fastai library is all focused on transfer learning
* network that does thing A, remove last layer or so, replace it with a few random layers at the end, fine tune those layers to do thing B, taking advantage of the features the original network learned
<img src="../../images/lesson8_transfer_learning.png" align="center"  height="300" width="500" >   

---

## Embeddings
embeddings allow us to use categorical data

<img src="../../images/lesson8_part1_2.png" align="center"  height="300" width="550" >   

