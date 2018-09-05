# Lesson 10:  NLP Classification and Translation
## (notes are continued)

(02-Apr-2018, live)  

### Part 2 / Lesson 10
- [Wiki Lesson 10](http://forums.fast.ai/t/part-2-lesson-10-wiki/14364)
- [Video Lesson 10](https://www.youtube.com/watch?v=h5Tz7gZT9Fo&feature=youtu.be) 
  - video length:  2:07:55
- http://course.fast.ai/lessons/lesson10.html
- Notebook:  
   * [imdb.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb)

---

## Language Model vs Word2Vec `58:20`
- one question that came up during break was: "How does this compare to word2vec?"
- this is actually a great thing for you to spend time thinking about during the week, is how does this compare to word2vec?
- I'll give you the summary now, but it's a very important conceptual difference
- The main conceptual difference is, what is word2vec?
- **word2vec** is a single embedding matrix.  Each word has a vector and that's it.  So, in other words, it's **a single layer** from a pre-trained model and, specifically, that layer is the input layer
- And also, specifically, that pre-trained model is a linear model, that is pre-trained on something that is called a **co-occurrence** matrix
- so we have no particular reason to believe that this model has learned anything much about the English language.  Or that it has any particular capabilities because it just has a single linear layer, and that's it.
- So, what's this wikitext 103 model? It's a language model.  It has a 400 dimensional embedding matrix, 3 hidden layers, with 1150 activations per layer and regularization, and all of that stuff.  
```python
em_sz, nh, nl = 400, 1150, 3
```
- tied in ?, input, output matrices, matrix equations ? , it's basically a state-of-the-art AWD LSTM
- ASGD Weight-Dropped == AWD
- ASGD = Asynchronous Stochastic Gradient Descent 
- so, what's the difference between a single layer of a single linear model vs a 3-layer recurrent neural network?
- 
