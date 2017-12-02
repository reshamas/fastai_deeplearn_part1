# Lesson 5

[Livestream: Lesson 5](https://www.youtube.com/watch?v=J99NV9Cr75I&feature=youtu.be)

[Wiki: Lesson 5](http://forums.fast.ai/t/wiki-lesson-5/8408)  

Notebooks:  [lesson5-movielens.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb)

## Blogs to Review

* [Structured Deep Learning](https://towardsdatascience.com/structured-deep-learning-b8ca4138b848) Kerem Turgutlu (Masters' student at USF)
  - experiment with different Kaggle datasets
  - experiment with dropout, etc
  - unexplored territory
* [Fun with small image data-sets (Part 2)](https://medium.com/@nikhil.b.k_13958/fun-with-small-image-data-sets-part-2-54d683ca8c96) Nikhil B
* [How do We Train Neural Networks?](https://towardsdatascience.com/how-do-we-train-neural-networks-edd985562b73) Vitaly Bushev
  - great technical communication

## Summary of Course so Far
### First Half of this Course
- getting through applications for use
- here's the code you have to write
- high level description of what code is doing 

### Second Half of this Course
- now, we're going in reverse; digging into detail
- dig into source code of fastai library and replicate it
- no more best practices to show us

### Today's Lesson
- create effective collaborative filtering model from scratch
- will use PyTorch as an automatic differentiating tool (won't use neural net features)
- will try not to use fastai library

## Collaborative Filtering
Movie Lens Dataset:  [lesson5-movielens.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb)

We will use 3 columns of data:  
- userId (categorica)
- movieId (categorical)
- rating (continous, dependent var)





