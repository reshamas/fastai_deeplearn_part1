# Lesson 9: Multi-object detection
(26-Mar-2018, live)  
 
- [Wiki: Part 2 / Lesson 9](http://forums.fast.ai/t/part-2-lesson-9-wiki/14028)
- [Lesson 9 video](https://www.youtube.com/watch?v=0frKXR-2PBY) 
  - video length:  2:05:33
- http://course.fast.ai/lessons/lesson9.html
- Notebook:  
   * [pascal.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/pascal.ipynb)

---

## Start Class
- today we will continue working on object detection, which means that for every object in a photo with 1 of 20 classes...
- we are going to figure out what the object is, and what its bounding box is such that we can apply that model to a new dataset with unlabeled data and add those labels to it
- general approach is to start simple and gradually make it more complicated; we started last week with a simple classifier, 3 lines of code, 
- we make it slightly more complex to turn it into a bounding box without a classifier
- today, we will put those two pieces together to make a classifier plus a bounding box 
- all of these are for a single object, the largest object in the image, and from there we will build up something which is closer to our final goal
- you should go back and make sure you understand all the concepts from last week

<img src="../../images/lesson_09/lesson9_know_these1.png" align="center"  height="400" width="600" >   
<br>

<img src="../../images/lesson_09/lesson9_know_these2.png" align="center"  height="400" width="600" >   



## Things to Know from Last Week
- Pathlib; JSON
- dictionary comprehensions
- `defaultdict`
- knowing how to jump around fastai source code is important
- lambda functions --> they come up everywhere
- matplotlib API
- Bounding box coordinates
- Custom head; bounding box regression --> will come up in every lesson

## What You Should Know from Part 1 of Course
- How to view model inputs from a DataLoader
- How to view model outputs
- 

<img src="../../images/lesson_09/lesson9_data_loader.png" align="center"  height="300" width="500" >   



## Today's Lesson
- we were working through Pascal notebook
