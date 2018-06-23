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
- today we will continue working on object detection, which means that for every object in a photo with 1 of 20 classes,
- we are going to figure out what the object is, and what its bounding box is such that we can apply that model to a new dataset with unlabeled data and add those labels to it
- general approach is to start simple and gradually make it more complicated; we started last week with a simple classifier, 3 lines of code, 
- we make it slightly more complex to turn it into 
