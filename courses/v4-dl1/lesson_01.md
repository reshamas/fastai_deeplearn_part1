# Lesson 1
- Live:  17-Mar-2020
- Time: 6:30 to 9pm PST  (9:30pm to midnight EST)
- course will be released in July
- supposed to be the official version (now **v4**)
- book: [Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD ](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527)

## Homework
- [Lesson 1 Homework](https://forums.fast.ai/t/did-you-do-the-homework/660340)

- [x] make sure you can spin up a GPU server
- [x] that you can shut it down when it is finished
- [x] run the code shown in the lecture
- [x] use the documentation, use the doc function inside juypter notebook
- [x] do some searching of the fast.ai docs
- [ ] see if you can grab the fast.ai documentation notebooks and try running them
- [ ] read a chapter of the fast.ai book 11
- [ ] do the questionnaire at the end of the chapter (not everything has been covered yet, answer only the questions that you can)
- [ ] try to get comfortable with running code

## Referencing Documentation in Jupyter Notebook

#### `?` OR `??` gives interactive python guide  (abbreviated output below. [full doc output](doc_01_reference.md))

```text
IPython -- An enhanced Interactive Python
=========================================

IPython offers a fully compatible replacement for the standard Python
interpreter, with convenient shell features, special commands, command
history mechanism and output results caching.

At your system command line, type 'ipython -h' to see the command line
options available. This document only describes interactive features.

GETTING HELP
------------

Within IPython you have various way to access help:

  ?         -> Introduction and overview of IPython's features (this screen).
  object?   -> Details about 'object'.
  object??  -> More detailed, verbose information about 'object'.
  %quickref -> Quick reference of all IPython specific syntax and magics.
  help      -> Access Python's own help system.

If you are in terminal IPython you can quit this screen by pressing `q`.
```

- `?learn` gives (`learn?` works too)
```bash
Signature:   learn(event_name)
Type:        Learner
String form: <fastai2.learner.Learner object at 0x7f5ffb61dfd0>
File:        /opt/conda/envs/fastai/lib/python3.7/site-packages/fastai2/learner.py
Docstring:   Group together a `model`, some `dls` and a `loss_func` to handle training
```
- `??learn` gives entire class info  (doc abbreviated here); (`learn??` works too)
```bash
Signature:   learn(event_name)
Type:        Learner
String form: <fastai2.learner.Learner object at 0x7f5ffb61dfd0>
File:        /opt/conda/envs/fastai/lib/python3.7/site-packages/fastai2/learner.py
Source:     
class Learner():
    def __init__(self, dls, model, loss_func=None, opt_func=Adam, lr=defaults.lr, splitter=trainable_params, cbs=None,
                 metrics=None, path=None, model_dir='models', wd=None, wd_bn_bias=False, train_bn=True,
                 moms=(0.95,0.85,0.95)):
        store_attr(self, "dls,model,opt_func,lr,splitter,model_dir,wd,wd_bn_bias,train_bn,metrics,moms")
        self.training,self.create_mbar,self.logger,self.opt,self.cbs = False,True,print,None,L()
        if loss_func is None:
            loss_func = getattr(dls.train_ds, 'loss_func', None)
            assert loss_func is not None, "Could not infer loss function from the data, please pass a loss function."
        self.loss_func = loss_func
        self.path = path if path is not None else getattr(dls, 'path', Path('.'))
        self.add_cbs([(cb() if isinstance(cb, type) else cb) for cb in L(defaults.callbacks)+L(cbs)])
        self.model.to(self.dls.device)
        if hasattr(self.model, 'reset'): self.model.reset()
        self.epoch,self.n_epoch,self.loss = 0,1,tensor(0.)

    @property
    def metrics(self): return self._metrics
    @metrics.setter
    def metrics(self,v): self._metrics = L(v).map(mk_metric)
```
- `?learn.predict` gives:
```bash
Signature: learn.predict(item, rm_type_tfms=None, with_input=False)
Docstring: Return the prediction on `item`, fully decoded, loss function decoded and probabilities
File:      /opt/conda/envs/fastai/lib/python3.7/site-packages/fastai2/learner.py
Type:      method
```

- `??learn.predict` gives:
```bash
Signature: learn.predict(item, rm_type_tfms=None, with_input=False)
Docstring: Return the prediction on `item`, fully decoded, loss function decoded and probabilities
Source:   
    def predict(self, item, rm_type_tfms=None, with_input=False):
        dl = self.dls.test_dl([item], rm_type_tfms=rm_type_tfms)
        inp,preds,_,dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
        dec = self.dls.decode_batch((*tuplify(inp),*tuplify(dec_preds)))[0]
        i = getattr(self.dls, 'n_inp', -1)
        dec_inp,dec_targ = map(detuplify, [dec[:i],dec[i:]])
        res = dec_targ,dec_preds[0],preds[0]
        if with_input: res = (dec_inp,) + res
        return res
File:      /opt/conda/envs/fastai/lib/python3.7/site-packages/fastai2/learner.py
Type:      method
```

- `doc(learn)` gives

```text
Learner object at 0x7f5ffb61dfd0>[source]
Learner object at 0x7f5ffb61dfd0>(event_name)

Group together a model, some dls and a loss_func to handle training
```

- `doc(learn.predict)` gives
```text
Learner.predict[source]
Learner.predict(item, rm_type_tfms=None, with_input=False)

Return the prediction on item, fully decoded, loss function decoded and probabilities

Show in docs
```
- `ImageDataLoaders` + <kbd> shift </kbd> + <kbd> tab </kbd> 
```text
Init signature: ImageDataLoaders(*loaders, path='.', device=None)
Docstring:      Basic wrapper around several `DataLoader`s with factory methods for computer vision problems
File:           /opt/conda/envs/fastai/lib/python3.7/site-packages/fastai2/vision/
```

## Paperspace
- fastai: [Getting Started with Gradient](https://course.fast.ai/start_gradient.html)
- fastai: v4 [Paperspace (free, paid options)](https://forums.fast.ai/t/platform-paperspace-free-paid-options/65515)

### My steps on Paperspace
1.  notebook:  https://www.paperspace.com/telmjtws3/notebook/prjrrhy56
2.  Open terminal, via Jupyter Notebook
- type `bash` to get a regular terminal (autocomplete, etc)
- `pip install fastai2 fastcore --upgrade`
- `cd course-v4`
- `git pull`


## Logistics
- edited video will be available in 1-2 days
- whatever you ask on the forum, it will eventually be public
- it's not personal if your post gets deleted from forums, it's for the readability of the forums
- 800 most valued members of community taking course
- at 9:40pm EST, there are **441** people watching
- at 9:45pm EST, **465**
- at 10:00pm EST, **483**
- at 111:45pm EST, **434**
- at 12am, **405**

## Forums
- can select "none" to remove study group threads
- study group:  research shows doing work in a group are much more likely to create powerful, long-term projects
- will set up virtual study groups

## COVID-19
- blog: [Covid-19, your community, and you — a data science perspective](https://www.fast.ai/2020/03/09/coronavirus/)
- published:  09-Mar-2020
- 1/2 million people read the blog
- post translated in 15 languages
- OPEN Forum category:  [covid-19](https://forums.fast.ai/c/covid-19/52)

10:33 break

## Getting Started
- AGI:  Artificial General Intelligence
- Neural networks:  a brief history
- 1986:  MIT released book on Parallel Distributed Processing (PDP)
- 

## Education at Bat: 7 Principals for Educators
Professor David Perkins uses his childhood baseball experiences:  
1. Play the whole game
2. Make the game worth playing
3. Work on the hard parts

You will be practicing things that are hard.  Requires:
- tenacity
- committment
- will need to work damn hard
- spend less time on theory, and MORE time on running models and with code

## Software Stack
- fastai
- PyTorch
- Python

## PyTorch
- Tensorflow got bogged down
- PyTorch was easier to use
- in last 12 months, % of papers that use PyTorch at conferences went from 20% to 80%
- industry moves slowly, but will catch up
- PyTorch:  very flexible, not designed for beginner-friendliness
  - doesn't have higher level libraries
  - fastai is the most popular higher level API for PyTorch
  - fastai uses a layered API
  
## To do work
- need GPU, Nvidia one
- use one of the platforms that is easily set up
- run it on Linux; it's hard enough to learn deep learning w/o archane solutions
- app_jupyter.ipynb:  learn about Jupyter notebook
- REPL:  Read, Evaluate, Print, Loop

## Jupyter Notebook
- shift + enter:  to run
- Workflow:  select notebook, duplicate it and rename it 
- fastbook repository: all text from book 
- course-v4 --> this removes all text, leaves just code
- at the end of notebooks, there are Questionnaires
  - What do we want you to take away from each notebook?
  - What should you know before you move on?
  - Do questionnaire before moving on to each chapter
  - If you missed something, do go back and read it
  - If you do get stuck after a couple of times, then do move on to next chapter and you might understand it better later
  - File / Trust Notebook 
- `jupyter labextension install @jupyter-widgets/jupyterlab-manager`

## 
- deep learning is a kind of machine learning
- 

## Limitations Inherent to Machine Learning
- 

## Consider how a model interacts with its environment
- PROXY:  arrest is a proxy for crime [listen to this again]
- 

## Homework
1. spin up a GPU server
2. run code
3. search fastai docs
4. try to get comfortable, know your way around
5. read chapter of book
6. go through questionnaire


