# Lesson 1

- Live Date:  22-Oct-2018
- [Wiki](https://forums.fast.ai/t/lesson-1-class-discussion-and-resources/27332)
- [Video](https://www.youtube.com/watch?v=BWWm4AzsdLk)
  - Video duration:  1:40:11
- Notebook:  
  - [lesson1-pets.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb)
- [fastai library](https://github.com/fastai/fastai)

---

## Homework:  parts completed ✅
- Google Cloud setup
  - Get your GPU going
- read lesson1-pets.ipynb notebook
- read [Tips for Building Image dataset](https://forums.fast.ai/t/tips-for-building-large-image-datasets/26688)
- read [Lesson 1 notes](https://forums.fast.ai/t/deep-learning-lesson-1-notes/27748)
- read fastai documentation
- run lesson1-pets.ipynb

## Homework: To Do
- get your own image dataset
  - Repeat process on your own dataset
  - Share on Forums
- repo:  fastai_docs
  - download repo
  - run the code
  - experiment
  - git/clone, open in Jupyter (GitHub doesn't render notebooks so well)
- Use the first notebook

## Lesson 1 Pets
```bash
RuntimeError: CUDA error: out of memory
```
Note:  reduce batch size and restart kernel



---

# Intro
- slightly delayed, waiting for students to get through security
- For in-class students in SF:
  - get to know your group of 6
  
## Pete Maker
- intro from PG&E
- USF specific site procedures (earthquake, emergencies, evacuation)

## [David Uminsky](https://www.linkedin.com/in/david-uminsky-5153b1a8/)
- Professor of DS at USF
- Diversity Fellows sponsored by:  EBay, Facebook
- 3rd iteration of this course (started from 60-80 students to 280)

## Rachel Thomas

## Jeremy Howard
- largest group of people joining:  Bangalore, India
- US towns
- Lagos

## Computer for in-class
1.  AWS Salamander
2.  AWS EC2
3.  Google Compute Platform (GCP)

## Computers for Int'l
1.  Google Computer Platform 
  - has fastai image
  - $300 credits
2.  AWS EC2 $0.90/hr  

## GCP
https://cloud.google.com

### Advice
Pick one project, do it very well, and make it fantastic.

doc(interp.plot_top_losses)
- prediction, actual, loss, probability it was predicted
- Don't be afraid to look at the source code.
- confusion matrix, if you have lots of classes, don't use confusion matrix.  use interp.most_confused. 
- `unfreeze`:  please train the whole model
- if you run out of memory, use a smaller batch size


