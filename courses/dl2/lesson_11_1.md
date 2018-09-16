# Lesson 11:  Neural Translation

(09-Apr-2018, live)  

- [Wiki Lesson 11](http://forums.fast.ai/t/part-2-lesson-11-wiki/14699)
- [Video Lesson 11](https://www.youtube.com/watch?v=tY0n9OT5_nA&feature=youtu.be) 
  - video length:  2:15:57
- http://course.fast.ai/lessons/lesson11.html
- Notebook:  
   * [imdb.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb)

---
# Lesson Description
Today we’re going to learn to **translate French into English**! To do so, we’ll learn how to add **attention to an LSTM** in order to build a sequence to sequence (seq2seq) model. But before we do, we’ll do a review of some key RNN foundations, since a solid understanding of those will be critical to understanding the rest of this lesson.

A seq2seq model is one where both the input and the output are sequences, and can be of difference lengths. Translation is a good example of a seq2seq task. Because each translated word can correspond to one or more words that could be anywhere in the source sentence, we learn an attention mechanism to figure out which words to focus on at each time step. We’ll also learn about some other tricks to improve seq2seq results, including teacher forcing and bidirectional models.

We finish the lesson by discussing the amazing DeVISE paper, which shows how we can bridge the divide between text and images, using them both in the same model!

---
## `00:00` Class Intro
- I want to start pointing out a couple of mini-cool things that happened this week.  One thing I'm really excited about is we briefly talked about how [Leslie Smith](https://twitter.com/lnsmith613) has a new paper out, and it basically, the paper goes, takes his previous two key papers (1: cyclical learning rates and 2: super convergence) and builds on them with a number of experiments to show how you can achieve super-convergence.  Super-convergence lets you train models 5 times faster than previous, kind of step-wise approaches.  It's not 5 times faster than CLR, but it's faster than CLR as well. And the key is that **super-convergence lets you get up to like massively high learning rates** by somewhere between 1 and 3 which is quite amazing.  
- And so the interesting thing about super-convergence is that it... You actually train at those very high learning rates for quite a large percentage of your epochs, and during that time, the loss doesn't really improve very much.  But the trick is it's doing a lot of searching through the space to find really generalizable areas, it seems.  
