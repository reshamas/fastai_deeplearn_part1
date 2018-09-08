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
- and we just grab that many rows, so from i to i+70 rows and then we try to predict that plus 1.  Remember, so we're trying to predict 1 past what we're up to.  So, we've got 64 columns and each of those is 1/64th of our 25 million tokens, you know, hundreds of thousands long, and we just grab, you know 70 at a time so each of those columns, each time we grab it, is going to hook out to the previous column.  And that's why we get this consistency.  This language model is **stateful** which is very important.
- Pretty much all the cool stuff in the language model is stolen from Steven Merity's AWD-LSTM, including this little trick here which is, if we always grab 70 at a time, and then we go back and do a new epoch, we're going to grab exactly the same batches every time -- there is no randomness.  
  - Reference: [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/pdf/1708.02182.pdf)
- Now, normally, we shuffle our data, every time we do an epoch or every time we grab some data, we grab it at random.  But, you can't do that with a language model because this set has to join up with the previous set because it's trying to learn the sentence, right? And if you suddently jump somewhere else, then that doesn't make any sense as a sentence
- So, Steven's idea is to say, ok, since we can't shuffle the order, let's instead randomly change the size, **the sequence length**.  So basically he says, 95% of the time we'll use bptt 70, *but* 5% of the time we will use half of that.  And then he says, you know what, I'm not even going to make that the sequence length, I'm going to create a normally distributed random number with that average and a standard deviation of 5, and I'll make that the sequence length, right? So the sequence length is 70'ish, and that means every time we go through, we're getting slightly different batches.  So, we've got that little bit of extra randomness.
- I asked Steven Merity where he came up with this idea, did he think of it? And, he was like, I think I thought of it, but it seemed so obvious that I bet I didn't think of it.  Which is like true of like every time I come up with an idea in deep learning, you know, it always seems so obvious that somebody else has thought of it, but I think he thought of it.
- So, yes, this is a nice thing to look at if you're trying to do something a bit unusual with the data loader, ok, here's a simple kind of raw (?role) model you can use as to creating a data loader from scratch, something that spits out batches of data

## Language Model Loader (back to notebook) `1:09:10`
- So our langauge model just took in all of the documents concatenated together, along with the batch size and bptt
```python
trn_dltrn_dl  ==  LanguageModelLoaderLanguage (np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)
```
#### Creating Your Own Learner and ModelData Classes is easy as this!
- Now, generally speaking, we want to create a learner and the way we normally do that is by getting a model data object and by calling some kind of method which have various names but sometimes, often we call that method `md.get_model`.  So the idea is that the model data object has enough information to know what kind of model to give you.  So, we have to create that model data object:  
`md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)` which means we need that class.  And that's very easy to do, right?
```python
learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)
```
- So, here are all of the pieces. We are going to create a custom learner, a custom model data class and a custom model class
- [nlp.py](https://github.com/fastai/fastai/blob/7ac2c490c22e2f0c0ffe983e593c4671d6beed2b/fastai/nlp.py)
- A model data class, again this one doesn't inherit from anything, so you really see... there is almost nothing to do. You need to tell it, most importantly: what's your training set, give it a data loader `trn_dl`, what's the validation set `val_dl`, give it a data loader, and optionally, give it a test set `test_dl1` data loader
- `01:10:22` Plus anything else it needs to know. So, it might need to know the `bptt`, it needs to know the number of tokens(`nt`, that's the vocab size), it needs to know what is the padding (`pad_idx`) index.  And so that it can save temporary files in models, model data's always need to know the path (`path`)
- and so we grab all of that stuff, and we dump it.  and that's it, that's the entire **initializer** (`__init__`), there's no logic there at all 
- so then, all of the work happens inside `get_model`.  And so, `get_model` calls something that we will look at later which just grabs a normal PyTorch nn.module architecture, ok? 
- and chucks it on the GPU:  `model = LanguageModel(to_gpu(m))`.  Note:  with PyTorch, normally we would say `.cuda()`.  With fast.ai, it's better to say `to_gpu`.  And the reason is that if you don't have a GPU, it will leave it on the CPU, and it also provides a global variable you can set to choose whether it goes on the GPU or not.  So, it's a better approach
- So, we wrap the model in a language model and the language model is `class LanguageModel()`
    - a Language Model is a sub-class of BasicModel
    - it almost basically does nothing except it defines layer groups.  And so, remember how when we do discriminative learning rates where different layers have different learning rates or like we freeze different amounts, we don't provide a different learning rate for every layer, because there can be 1000 layers, **we provide a learning rate for EVERY LAYER GROUP**.  So, when you create a custom model, you just have to overwrite this one thing which returns a list of all of your layer groups.
    - So, in this case, my last layer group contains the last part of the model and one bit of dropout:  `(self.model[1], m.dropouti)`
    - And, the rest of it, this * here, means pull this apart, so this going to be 1 layer per RNN layer: `*zip(m.rnns, m.dropouths)`
    - So, that's all that is 
```python
class LanguageModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [*zip(m.rnns, m.dropouths), (self.model[1], m.dropouti)]
        
class LanguageModelData():
    def __init__(self, path, pad_idx, nt, trn_dl, val_dl, test_dl=None, bptt=70, backwards=False, **kwargs):
        self.path, self.pad_idx, self.nt = path, pad_idx, nt
        self.trn_dl, self.val_dl, self.test_dl = trn_dl, val_dl, test_dl
    
    def get_model(self, opt_fn, emb_sz, n_hid, n_layers, **kwargs):
        m = get_language_model(self.nt, emb_sz, n_hid, n_layers, self.pad_idx, **kwargs)
        model = LanguageModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=optn_fn)

class RNN_Leaner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.cross_entropy
```
- And then, finally, turn that into a **learner**, and so a learner, you just pass in the model, and it turns it into a learner! `return RNN_Learner(self, model, opt_fn=optn_fn)`
- In this case, we have overwritten learner; and the only thing we've done is to say, I want the default loss function to be cross-entropy
- So this entire set of custom Model, custom Model Data, custom (RNN) Learner, all fits on a single screen, and they always basically look like this
- So, that's a kind of dig inside this boring part of the code base
- the **interesting** part of this **code base** is **get_language_model**.  Because `get_language_model` is the thing that gives us our **AWD LSTM** and it actually contains the big idea -- the big incredibly simple idea that everybody else here thinks is really obvious.  But everyone in the NLP community I spoke to thought was insane, which is basically, every model can be thought of... pretty much every model can be thought of as a backbone plus a head -- and if you pre-train the backbone and stick on a random head, you can do fine tuning and that's a good idea, right?  
- So, here are 2 bits of code right here, literally right next to each other, this is kind of all there is inside of  
- 

