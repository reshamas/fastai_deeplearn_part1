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
- `1:00:15` Everything!  You know, they're very different levels of capability.  You will see when you try using a pre-trained language model vs a word2vec layer, you'll get very, very different results, for the vast majority of tasks
- Rachel:  what if the numpy array does not fit in memory?  Is it possible to write a PyTorch data loader directly from a large csv file?
- JH: It almost certainly won't come up, so I won't spend time on it.  These things are tiny, they're just ints, think about how many ints you would need to run out of memory.  It's not going to happen.  They don't have to fit in GPU memory, just in your memory.
- So, I've actually done another Wikipedia model, which I called Giga Wiki which is on *all* of Wikipedia and even that easily fits in there 
- The reason I'm not using it [Giga Wiki] is because it turned out not to really help very much vs wikitext 103, but I've built a bigger model than anyone else I've found in the academic literature and it fits in memory on a single machine.  
- Rachel:  What is the idea behind averaging the weights of the embeddings?
- JH:  They're going to be set to something, you know.  There are words that weren't there, so other options is we could leave them at 0, but that seems like a very extreme thing to do, like zero is a *very extreme number*.  Why would it be 0? We could set it equal to some random numbers, but if so, what would be the mean and standard deviation of those random numbers? Or should they be uniform? If we just average the rest of the embeddings, than we have something that's a reasonable scale.
- Rachel:  Just to clarify, this is how you're initializing words that didn't appear in the training?
- JH:  Yes, that's right
- Rachel: And, then, I think you've pretty much answered this one, but someone had asked if there is a specific advantage to creating our own pre-trained embedding over using glob or word2vec? 
- JH: Yes, I think we have.  We're not creating a pre-trained embedding, we're creating a pre-trained model

## Language Model `1:02:20`
- ok, let's talk a little bit more... there's a ton of stuff you've seen before, but actually 
