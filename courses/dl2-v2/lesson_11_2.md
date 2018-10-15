# Lesson 11_2:  Neural Translation

(09-Apr-2018, live)  

- [Wiki Lesson 11](http://forums.fast.ai/t/part-2-lesson-11-wiki/14699)
- [Video Lesson 11](https://www.youtube.com/watch?v=tY0n9OT5_nA&feature=youtu.be) 
  - video length:  2:15:57
- http://course.fast.ai/lessons/lesson11.html
- Notebook:  
   * [translate.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/translate.ipynb)
   
## `01:13:10` 
- So one question that came up during the break is that some of the tokens that are missing in fast text like had a curly quote rather than a straight quote, for example.  And the question was, would it help to normalize punctuation?  And, the answer for this particular case is, probably yes, the difference between curly quotes and straight quotes is really semantic.  You do have to be very careful, though, because like it may turn out that people using beautiful curly quotes are like using more formal language and they're actually writing in a different way so I generally... you know, if you're going to do some kind of pre-processing like punctuation normalization, you should definitely check your results with and without because like *nearly always* that kind of pre-processing makes things worse even when I'm sure it won't.

### `01:14:10`  Question
- Person x (Yannet?):  Hello, what might be some ways of realizing? these sequence to sequence models besides dropout and weight decay?
- JH:  Let me think about that during the week. 
