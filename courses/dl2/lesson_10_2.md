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
- ok, let's talk a little bit more... there's a ton of stuff you've seen before, but it's changed a little bit, it's actually a lot easier than it was in Part 1, but I want to go a little bit deeper into the language model loader.  
- on slide:  "You can use any iterable that creates a stream of batches as a data loader"
- So, this is the language model loader.  I really hope by now you've learned in your editor or IDE how to jump to symbols.  I don't want it to be a burden for you to find out what the source code of language model loader is, alright? 
- https://github.com/fastai/fastai/blob/7ac2c490c22e2f0c0ffe983e593c4671d6beed2b/fastai/nlp.py
- and, if it's a burden, please go back and try to learn those keyboard shortcuts in VS Code
- you know, if your editor doesn't make it easy, don't use that editor anymore, ok?  There are lots of good free editors that make this easy
- Here's the source code for `LanguageModelLoader`.  It's interesting to note that it is not doing anything particularly tricky.  It's not deriving from anything at all.  What makes it something that is capable of being a data loader is that it's something that you can iterate over.  
```python
class LanguageModelLoader():

    def __init__(self, ds, bs, bptt, backwards=False):
        self.bs,self.bptt,self.backwards = bs,bptt,backwards
        text = sum([o.text for o in ds], [])
        fld = ds.fields['text']
        nums = fld.numericalize([text],device=None if torch.cuda.is_available() else -1)
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        return self

    def __len__(self): return self.n // self.bptt - 1

    def __next__(self):
        if self.i >= self.n-1 or self.iter>=len(self): raise StopIteration
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        res = self.get_batch(self.i, seq_len)
        self.i += seq_len
        self.iter += 1
        return res

    def batchify(self, data):
        nb = data.size(0) // self.bs
        data = data[:nb*self.bs]
        data = data.view(self.bs, -1).t().contiguous()
        if self.backwards: data=flip_tensor(data, 0)
        return to_gpu(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)
```
- `1:03:40` So, specifically, here is the `fit` function inside fastai.model.  This is where everything ends up eventually which goes through each epoch, and then it creates an iterator from the data loader and then just does for-loop through it
- So anything you can do a for-loop through, can be a data loader.  Specifically, it needs to return tuples of mini-batches, independent and dependent variables, for mini-batches.  
- Anything with a ?dunda iter method (`__iter__`), is something that can act as an iterator, and `yield` is a neat little Python keyword you should probably learn about if you don't already know it, but it basically spits out *a thing* and waits for you to ask for another thing, normally in a for-loop or something
- [Python yield keyword explained](https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/)
- So, in this case, we start by initializing the language model, passing it in the numbers.  So, this is the numericalized, big long list of all of our documents concatenated together
- and the first thing we do is to "batchify" it.  And this is the thing **quite a few of you got confused** :heavy_exclamation_mark: about last time, right?  If our batch size is 64 and we have 25 million numbers in our list, we are not creating items of length 64.  We're not doing that.  We are creating 64 items in total.  So, each of them is size t/64 (t divided by 64), which is 390,000 (or 390,590 to be exact)
- That's what we do here when we reshape it (`def batchify`) so this axis here `data = data.reshape(self.bs, -1).T` is of length 64 and this `-1` is everything else, so that's 390,590 blob, and then we transpose it
- That means we now have **64 columns**, **390,590 rows** and then what we do each time we do an iterate is we grab one batch of some sequence length (we'll look at that in a moment), but basically, it's approximately equal to `bptt` which we set to 70.  **`bptt` = back prop through time**
- 
