# Lesson 2:  Computer Vision - Deeper Applications

- Live Date:  30-Oct-2018
- [Lesson 2: official resources and updates](https://forums.fast.ai/t/lesson-2-official-resources-and-updates/28630)
- [Video](https://www.youtube.com/watch?v=BJFOXf_PrkA)
  - Video duration:  1:58:47
- Notebook:  
  - [lesson2-download.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb)
  - [lesson2-sgd.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-sgd.ipynb)
- [fastai library](https://github.com/fastai/fastai)
- [Lesson 2 Notes](https://forums.fast.ai/t/deep-learning-lesson-2-notes/28772)
- [links to different parts in the video](https://forums.fast.ai/t/lesson-2-links-to-different-parts-in-video/28777)

---

## Timeline
### `0:00` Discourse Overview

### `3:18`  Update Repositories
- https://course-v3.fast.ai
- Select `Returning to work`
- select the platform you are using
- only use the commands here

#### Update the course repo (Jupyter notebooks)
```bash
cd tutorials/fastai
git pull
```
#### Update the fastai library (Python libraries)
```bash
sudo /opt/anaconda3/bin/conda install -c fastai fastai
```

### `4:20` Share your work here
- https://forums.fast.ai/t/share-your-work-here/27676/84
- contributing to fastai libraries
- cool classifiers people built
  - Trinidad and Tobago Masquereders vs Regular Islanders
  - zucchini vs cucumber
  - Henri Palacci:  city maps of aerial satellite views

### `12:55` Next Steps
1.  Computer vision applications
2.  NLP applications
3.  Tabular applications
4.  Collaborative filtering applications
5.  Embeddings
6.  Computer vision deeper dive
7.  NLP deeper dive

- good for [people who are] learning to see things multiple times

#### If you're stuck, keep going
David Perkins, Harvard, Learning Theory:  
1.  Code first:  Focus on learning from experiments
2.  The whole game:  It's like learning soccer as a kid (Perkins)
3.  Concepts, not details:  We'll gradually dig into all the details
4.  Do Lesson 2:  ... even if you don't understand all of Lesson 1

### `16:20` 
- [lesson2-download.ipynb](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb)
- how to create your own classifier with your own images
- Jeremy's bear example
- this will download images to your server. It will use multiple processes:  `download_images(path/file, dest, max_pics=200)`
- One problem there is if something goes wrong, it's a bit hard to see what went wrong.  
- for `download_images`, `max_workers=0` that will do it without spinning up a bunch of processes and will tell you the errors better. 
- if things aren't downloading, try using `max_workers=0`
- remove images that aren't images at all; there are always a few that are corrupted, for whatever reason. 

#### `23:50` train/valid/test datasets
- train/valid/test --> often folders from Kaggle
- In this case, we don't have a separate validation set, because we just grabbed these images from Google Search.  But you still need a validation set.  Otherwise, you don't know how well your model is going.   
- If you don't have a separate training and validation set, you can just say that the training dataset is in the current folder `train="."`, and I want you to set aside 20% of the data, please:  `valid_pct=0.2`.  
- So, this is going to create a validation set for you automatically and randomly.  You'll see that whenever I create a validation set randomly, I always set my random seed to something fixed beforehand.  This means that every time I run this code, I will get the same validation set.  So, in general, I am not a fan of making my machine learning experiments reproducible by ensuring I get the same results every time.  The randomness to me, is really important.  The important part of finding out, is your solution stable, is it going to work each time you run it?  **But, what is important** is you have the same validation set.  Right, otherwise, when you are trying to to decide, has this hyperparameter change improved my model?  But, if you've got a different set of data you're testing it on, then you don't know, maybe that set of data just happens to be a bit easier, right.  So, that's why I always set the random seed here:  `np.random.seed(42)`
```python
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)     
  
data.show_batch(rows=3, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)        
```       
- So, we've now got a `DataBunch`.  You can now look inside at the `data.classes`, and you'll see these are the photos we created:  `['black','grizzly','teddys']`
- We can run `data.show_batch(rows=3, figsize=(7, 8))` and all the possible labels
- It tells us straight away that some of these are going to be a little tricky.  
- `data.c` tells us how many possible labels there are. We'll learn about more specific meanings of `c` later
- We can see how many things are in our training set:  `len(data.train_ds)`
- We can see how many things are in our validation set:  `len(data.valid_ds)`
- So, at that point, we can create our CNN.  
```python
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
```
- Combining a human expert with a learner is a good idea.  Very few people publish on this, few people teach this.  But, for me, it's the most useful skill.  Particularly, for you, you know most of the people watching this, are domain experts.  Not computer science experts.  This is where you can use your knowledge of point mutations in genomics, or Panamanian buses, or whatever.  So, let's see how that would work.  
- Do you remember the `plot_top_losses` from last time which are the images I am the most wrong about, or least confident about?  We're going to look at those and decide which of those are going to be noisy.  If you think about it, it's very unlikely that if there is some mis-labeled data, that it's going to be predicted correctly and with high confidence.  That's really unlikely to happen.  So, we're going to focus on the ones which the model is saying, either it's not confident of, or it was confident of, but it was wrong about.  They are the things which might be mis-labeled.  

## `31:30` Cleaning up:  Widgets in Jupyter Notebook 
- big shout-out to the San Francisco fastai study group who created this new widget this week, called the `FileDeleter`.  So, that's Zach, Jason and Francisco built this thing where we can basically take the `top_losses` from that interpretation object we just created.  And then, what we're going to do is say, ...
- if you don't pass it anything at all, it will return the entire dataset, and sort it.  The first thing will be the highest losses. 

```python
from fastai.widgets import *

losses,idxs = interp.top_losses()
top_loss_paths = data.valid_ds.x[idxs]
```
- So, this `top_loss_paths` contains all of the files in our dataset.  When I say our dataset, this particular one is in our validation set.  So, what this is going to do is it is going to clean up mis-labeled images or images that shouldn't be there.  And we're going to remove them from our validation set, so that our metrics will be more correct.  You then need to re-run these two steps, replacing `valid_ds` with `train_ds` to clean up your training set to get the noise out of that as well.  That's a good practice to do both.  
```python
losses,idxs = interp.top_losses()
top_loss_paths = data.train_ds.x[idxs]
```
- When we talk about test sets later as well, if you also have a test set as well, you can repeat the same thing.  So, we run `FileDeleter` passing in that list of paths:  
```python
fd = FileDeleter(file_paths=top_loss_paths)
```
- These are the ones it's either wrong about, or least confident about, and so, not surprisingly, this one here [woman with guitar] does not appear to be a teddy bear or a black bear or a brown bear, right. So, this shouldn't be in our dataset.  
- What I do is I whack on the `Delete` button [on the widget] and all the rest do indeed look like bears so I can click `Confirm` and it will bring up another 5.
- So what I'll do when I do this, is I'll keep going until I get to a couple of screen full of things that look ok, and that suggests to me that I've kind of got past the worse bits of the data.  
- And so, that's it.  I want you to do it for the training set as well and re-train your model.  
- So, I'll just note here that what our San Francisco group did here was that they actually built a little app *inside* Jupyter notebook which you might not have realized was possible.  But, not only is it possible, it's also *surprisingly* straightforward.  And, just like everything else, you can hit double question mark `??FileDeleter` to find out their secret.  So, here is the source code:  
[file_picker.py](https://github.com/fastai/fastai/blob/a5e32c8bb079a62fdbbcd82d5fd7d46750a566bb/fastai/widgets/file_picker.py)
- And, really, if you've done any GUI programming before, it will look incredibly normal.  There are basically callbacks for what happens when you click on a button.  
- ...
- This idea of creating applications inside notebooks, it's really underused!  But, it's super neat because it lets you create tools for your fellow practioners, for your fellow experimenters.  And you can definitely envision taking this a lot further.
- In fact, by the time your're watching this on the MOOC, you'll find that there's a whole lot more buttons here.  We've got a long list of to do's that we're going to add to this particular thing.  
- ...
- Now that you know what's possible to write applications in your notebook, what are you going to write?  And if you google for [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/), you can learn about the little GUI framework to find out what kind of widgets you can create, what they look like, and how they work, and so forth.  And you'll find, you know, it's actually a pretty complete GUI programming environment you can play with.  And this will all work nicely with your models and so forth.  It's not a great way to productionize an application, because it is sitting inside a notebook.  This is really for things which are going to help other practioners, other experimentalists and so forth.  For productionizing things, you need to actually build a production web app.  Which we'll look at next.  

---

## `37:35` Putting Your Model in Production
- So, after you have cleaned up your noisy images, you can then retrain your model and hopefully, you'll find it's a little bit more accurate.
- One thing you might be interested to discover when you do this, is it actually doesn't matter most of the time very much.  Now, on the whole, these models are pretty good at dealing with *moderate* amounts of noisy data.  The problem would occur is if your data was not *randomly* noisy, but *biased* noisy.  So, I guess the main thing I'm saying is if you go through this process of cleaning up your data and then you rerun your model and find that it's like 0.001% better, that's normal.  It's fine, but it's still a good idea to make sure that you don't have too much noise in your data in case it is biased.  So, at this point, we're ready to put our model in production and this is where I hear a lot of people ask me about, you know, which mega-Google, Facebook, highly distributed serving system they should use and how do they use a thousand GPUs at the same time and whatever else.  

- For the vast, vast, vast majority of things that you all do, you will want to actually run in production on **a CPU**, **not a GPU**.  Why is that?  Because a GPU is good at doing lots of things at the same time.  But, unless you have a very busy webiste, it's pretty unlikely that you're going to have 64 images to classify at the same time to put into a batch, into a GPU.  And, if you did, you've got to deal with all that queuing and running it all together.  All of your users have to wait until that batch has got to get filled up and run.  It's a whole lot of hassle, right.  And then if you want to scale that, there's another whole lot of hassle.  It's much easier if you just wrap one thing, throw it at a CPU to get it done, and it comes back again.  So, yes, it's going to take, you know, maybe 10 or 20 times longer, right.  So, maybe it'll take 0.2 seconds rather than 0.01 seconds.  That's about the kind of times we're talking about.  But, it's so easy to scale, right.  You can chuck it on any standard serving infrastructure.  It's going to be cheap as hell.  You can horizontally scale it really easily, okay.  So, most people I know who are running apps that aren't kind of at Google scale, based on deep learning, **are using CPUs**.  

- `40:05` And the term we use is **inference**, right.  So when you're running... when you're not training a model, but you've got a trained model and you're getting to predict things, we call that **inference.**.
- So, that's why we say here:
>You probably want to use CPU for inference, except at massive scale (and you almost certainly don't need to train in real-time).  If you don't have a GPU that happens automatically.  You can test your model on CPU like so:  
```python
# fastai.defaults.device = torch.device('cpu')
```
- So, at inference time, you've got your pre-trained model, you saved those weights, and how are you going to use them to create something like Simon Willison't cougar detection?  Well, first thing you're going to need to know is, what were the classes that you trained with.  Right, you need to not know not just what are they?  But what were the order?  So, you will actually need to like serialize that or just type them in or in some way make sure you've got exactly the same classes that you trained with.  
```python
data.classes
```
```bash
['black', 'grizzly', 'teddys']
```
- If you don't have a GPU on your server, it will use the CPU automatically.  If you want to test if you have a GPU machine, and you want to test using a CPU, you can just uncomment this line, and that tells fastai that you want to use CPU by passing it back to PyTorch.  
```python
# fastai.defaults.device = torch.device('cpu')
```

## `41:15` Code for Prediction with New Image
- So, here's an example... we're not... we don't have a cougar detector.  We have a teddy bear detector.  And my daughter Claire is about to decide whether to cuddle this friend [photo of black grizzly bear].  So, what she does, is she takes Daddy's deep learning model and she gets a picture of this. And here's a picture that she's uploaded to the web app, okay.  And here's a picture of the potentially cuddlesome object.  
- And, so we're going to store that in a variable called `img`.  So, `open_image` is how you open an image in fastai, funnily enough.  
```python
img = open_image(path/'black'/'00000021.jpg')
img
```
- Here is that list of classes that we saved earlier.  And so as per usual, we created a `DataBunch`.  But, this time, we're not going to create a data bunch from a folder full of images.  We're going to create a special kind of `ImageDataBunch` which is one that is going to grab one single image at a time.  So, we're not actually passing it any data.  The only reason we pass it a `path` is so that it knows where to load our model from, right.  That's just the path, that's the folder that the model is going to be in.  But, what we do need to do is that we need to pass it the same information that we trained with.  So, the same `transforms`, the same `size`, the same normalization (`normalize`).  This is all stuff we'll learn more about.  But, just make sure it's the same stuff that you used before.  And so, now you've got a data bunch that **actually doesn't have any data in it at all**.  It's just something that knows how to transform a new image in the same way that you trained with so that you can now do **inference.** 
```python
classes = ['black', 'grizzly', 'teddys']
data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data2, models.resnet34)
learn.load('stage-2')
```

- So, you can now create a CNN with this kind of fake data bunch.  And, again, you would use exactly the same model that you trained with.  You can now load in those saved weights, okay.  And so this is the stuff that you do once, just once when your web app is starting up, okay.  And it take you, you know, 0.1 of a second to run this code.  And then, you just go `learn.predict(img)` and it's lucky we did that because it **is not a teddy bear!**  This is actually a black bear.  
```python
pred_class,pred_idx,outputs = learn.predict(img)
pred_class
```
output:  
```bash
'black'
```
- So, thankfully, due to this excellent deep learning model, my daughter will avoid having a very embarrassing black bear cuddle incident. - `43:25` So, what does this look like in production?   

## `43:25` What does prediction of image look like in production?
- Well, I took [Simon Willison (co-creator of Django)](https://twitter.com/simonw)'s code and shamelessly stole it, made it probably a little bit worse and... but, basically, it's going to look something like this.
- So Simon used a really cool web app toolkit called [Starlette](https://www.starlette.io).   
- If you've ever used [Flask](http://flask.pocoo.org), this will look extremely similar, but but it's kind of a more *modern* approach.  By modern, what I really mean is you can use `await`.  It basically means that you can wait for something that takes a while,  such as grabbing some data, without using up a process. So, for things like "I want to get a prediction" or "I want to load up some data" or whatever, it's really great to be able to use this modern Python 3 asynchronous stuff.  So, Starlett would come highly recommended for creating your web app.  And so, yeah, you just:  
- create a `route` as per usual in a web app.  `@app.route("/classify-url", methods=["GET"])`
- And in that, you say this is `async` to ensure that it doesn't steal the process while it's waiting for things 
- you open your image:   `img = open_image(BytesIO(bytes))`
- you call `.predict(img)` 
- and you return that response: `return JSONResponse({`....
- and then you can use a, you know, whatever, a Javascript client or whatever to show it.  
- and that's it.  that's basically the main contents of your web app.  So, give it a go, you know, this week.  Even if you've never created a web application before, there's a lot of, you know, nice little tutorials online and kind of starter code, you know.  If in doubt, why don't you try starlette.  
- there's a free hosting you can use.  There's one called [Python Anywhere](https://www.pythonanywhere.com/) , for example.  The one that Simon's used, we'll mention that on the Forums.  It's something basically you can package on a Docker thing and shoot it off, and it'll serve it up for you. So, it doesn't even need to cost you any money.  
- **And, so, all these classifiers you're creating, you can turn them into web apps!**
- So, I'll be really interested to see what you're able to make of that.  That will be really fun.  
- `45:44` Okay, so let's take a break.  We'll come back at 7:35.  See you then.  
```python
@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(cat_learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })
```

## `45:55` Things that can go wrong
- [after break]
```text
- Most of the time things will train fine with defaults
- There's not much you really need to tune (despite what you've heard!)
- Most likely are:
  - Learning rate
  - Number of epochs
```
- So, let's move on.  So, I mentioned that, most of the time, the kind of rules of thumb I've shown you will probably work.  And if you look at the [share your work thread](https://forums.fast.ai/t/share-your-work-here/27676), you'll find most of the time people are posting things saying:  I downloaded these images, I tried this thing, they worked much better than I expected.  Well, that's cool.  
- And then, like 1 out of 20, says like, *ah, I had a problem.*  So, let's have a talk about what happens when you have a problem.  And this is where we're starting to get into a little bit of theory.  Because in order to understand *why* we have these problems, and how we fix them, it really helps to know a little about what's going on.  So, first of all, let's look at examples of some problems.  
- The problems basically will be... either:
  - your **learning rate** is too high or low, or
  - your **number of epochs** is too high or low.
  
### Learning rate (LR) too high 
- So, we're going to learn about what those mean and why they matter, but first of all, because we're experimentalists, let's *try* them, all right.  So, let's grow with our teddy bear detector and let's make our learning rate really high.  The default learning rate is 0.003 that works most of the time.  So, what if we try a learning rate of 0.5.  That's huge!  What happens?  Our **validation loss gets pretty damn high**.  Remember, this is normally something that's underneath 1.0, right.  So, if you see your validation loss do that, right, before we even learn what validation loss is, just know this.  If it does that, **your learning rate is too high.**  That's all you need to know, okay.  
  - Make it [LR] lower.
  - It doesn't matter how many epochs you do.  And if this happens, there's no way to undo this.  You have to go back and create your neural net again and fit from scratch with a lower learning rate.  
- So, that's learning rate too high.
```python
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(1, max_lr=0.5)
```

### `48:00` Learning rate (LR) too low 
- What if we use a learning rate not of 0.003, but "1 e neg 5"=`1e-5`, so 0.00001, right.  So, this is just...I've just copied and pasted what happened when we trained before with a default error, right, with our default learning rate.  And within one epoch, we were down to a 2 or 3 percent error rate.  With this really low learning rate, our error rate does get better, but *very, very slowly*, right.  And you can plot it.  If you go learn... to `learn.recorder` is an object which is going to keep track of lots of things happening while you train.  You can call `learn.recorder.plot_losses()` to print... to plot out the validation and training loss.  And you can just see them just like gradually going down so slow, right.  So, if you see that happening, then you have a learning rate which is too small, okay.
- So, bump it up by 10, or bump it up by 100 and try again.  
- The other thing you'll see **if your learning rate is too small** is that your **training loss will be higher than your validation loss.***  You never want a model where your training loss is higher than your validation loss.  That always means **you haven't fitted enough**.  Which means either your learning rate is too low, or your number of epochs is too low.  So, if you have a model like that, train it some more or... TRAIN IT with a HIGHER LEARNING RATE.  


### `49:40` Too few epochs
- So, what if we train for just one epoch.  Our error rate certainly is better than random, 5%.  But, look at this, the difference between training loss and validation loss.  A training loss is much higher than the validation loss.  So, too few epochs and too low a learning rate look very similar, right.  So, you can just try running more epochs.  And, if it's taking forever, you can try a higher learning rate.  Where we try a higher learning rate, and the loss goes off to 100,000 million, then put it back to where it was and try a few more epochs.  
- That's the balance, right.  That's basically all you care about 99% of the time.  And this is *only* the 1 in 20 times that the defaults don't work for you.

### `50:30` Too many epochs
- Too many epochs, we're going to be talking more about this, creates something called **overfitting**.  If you train for too long, as we're going to learn about, it will learn to recognize your particular teddy bears, but not teddy bears in general.  Here's the thing:  despite what you may have heard, **it's very hard to overfit with deep learning.** So, we were trying today to show you an example of overfitting, and I turned off everything.  I turned... and we're going to learn all about these terms soon...:
  - I turned up all the data augmentation.  
  - I turned off dropout.  
  - I turned off weight decay.  
  - I tried to make it overfit as much as I can.  
  - I trained it on a smallish learning rate.  
  - I trained it for a really long time.  
- And, like, maybe I started to get it to overfit... maybe.  But... so the only thing that tells you that you're overfitting is that the error rate improves for a while, and then starts getting worse again.  
- You will see a LOT of people, even people that claim to understand machine learning, tell you that if your training loss is lower than your validation loss, then you are overfitting.  As you will learn today in more detail and during the rest of the course, that is absolutely not true!  **Any model that is trained correctly will always have trained loss lower than validation loss.  That is not a sign of overfitting.  That is not a sign you've done something wrong.  That is a sign that you have done *something right*, okay.  
- The sign that you are overfitting is that your error starts getting worse.  Because that's what you care about, right.**  You want your model to have a low error.  So, as long as you're training, and your model error is improving, **you are not overfitting.**  How could you be?  

### `52:20`
- So, there's basically the four possible... they are the main four things that can go wrong.  There are some other details that we will learn about during the rest of this course.  But, honestly, if you stopped listening now, (please don't, that would be embarrassing).  And you just, like, okay, and then I'm going to download images, I'm going to create CNNs with resnet34 or resnet50, I'm going to make sure that my learning rate and number of epochs is okay.  And then, I'm going to chuck them up in a Starlette web API.  
- Most of the time, you're done.  At least for computer vision.  
- Hopefully, you'll stick around because you want to learn about NLP and collaborative filtering and tabular data and segmentation.  And stuff like that as well.  
 
### `53:10` Don't be afraid of math
- Let's now understand what's actually going on.  What does it mean... **loss** mean?  What does **an epoch** mean?  What is **learning rate** mean?  Because for you to really understand these ideas, you need to know what's going on.  And so, we're going to go all the way to the other side. 
- Rather than creating a state-of-the-art cougar detector, we're going to go back and create the simplest possible linear model.  So, we're going to actually start seeing... we're actually going to start seeing a little bit of math, okay.  But, don't be turned off.  It's okay, right.  We're going to do a little of math, but it's going to be totally fine. Even if maths is not your thing.  Because the first thing you're going to realize is that when we see a picture, like this number "8", it's actually just a bunch of numbers.  It's a matrix of numbers.  For this grayscale one, it's a matrix of numbers. 
- If it was a color image, it would be... have a third dimension.  So, when you add an extra dimension, we call it a **tensor** rather than a matrix.  It would be 3D tensor of numbers:  red, green and blue.  
- So when we created that teddy bear detector, what we actually did was we created a mathemetical function that took the numbers from the images of the teddy bears.  And the mathematical function converted those numbers into, in our case, three numbers:
  - A number for the probability that it is a teddy bear 
  - A probability that it is a grizzly
  - And a probability it is a black bear
- In this case, there's some hypothetical function that's taking the pixels representing a handwritten digit and returning 10 numbers: 
  - the probability for each possible outcome:  the numbers from 0 to 9 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
- And so what you'll often see, in our code and other deep learning code, is that you're... you'll find this: a bunch of probabilities, and then you'll find something called "Max" or "argmax" attached to it, a function called.  And so what that function is doing, is it's saying, "find the highest number, so the highest probability, and tell me what the index is.  So, `np.argmax` or `torch.argmax` of this array [numbers from 0 to 9 and each of their probabilities] would return this number here [8:  0.59].  We return index "8".  That makes sense?  
- In fact, let's try it. So, we know that the function to predict something is called `learn.predict`.  So, we can chuck two question marks before it, or after it, to get the **source code**:  
  - `??learn.predict`
  - `learn.predict??`
- And, here it is, right.  `pred_max=res.argmax()`
- And then, what is the class?  We just pass that into the `classes` array.  You should find that the source code in the fastai library can both kind of strengthen your understanding of the concepts and make sure that you know what's going on.  And really help you here.   

###  `56:30` Using the fastai Documentation
#### Questions from class 
- Jeremy (JH):  You've got a question, come on over.
- Rachel (RT):  Can we have a definition of the error rate being discussed and how it is calculated?  I assume it's cross validation error.
- JH:  Sure.  So one way to answer the question of how is error rate calculated would be to type `error_rate??` and look at the source code, and it is:  `return 1-accuracy(input, targs)`.  Fair enough.  
- And so then a question might be "What is accuracy?"  Type `accuracy??`.  It is in [metrics.py](https://github.com/fastai/fastai/blob/bb3710a549fd28658301b6cb3e558b1e684a8bd8/fastai/metrics.py), which is at: `fastai/fastai/metrics.py`.  It is `argmax`.  So we now know that means find out which particular thing it is.  And then look at how often that equals the target `(input==targs)`.  So, in other words, the actual value.  And take the mean.  So, that's basically what it is.
```python
def accuracy(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    return (input==targs).float().mean()
```
- And so the question is, okay, well what is that being applied to?  And always in fastai, metrics, so these things that we pass in [`metrics=error_rate` in `learn = create_cnn(data, models.resnet50, metrics=error_rate, ps=0, wd=0)`], we call them `metrics`.  Metrics are always going to be applied to the **validation set**, okay.  So, anytime you put a metric here, it will be applied to the validation set because that's your best practice, right.  That's like, that's what you always want to do... is make sure that you're checking your performance on data that your model hasn't seen.  And we'll be learning more about the validation set shortly.  
- Remember, you can also type `doc(accuracy)`.  If the source code is not what you want, which it might well not be, you actually want **The Documentation**, that will both give you a summary of the types in and out of the function **and a link to the full documentation!**  where you can find out all about how metrics work and what other metrics there are and so forth.  And, generally speaking, you'll also find links to more information where, for example, you will find complete runs through with sample code and so forth showing you how to use all these things.  So, don't forget that the `doc` function is your friend, okay.
- And, also in the documentation, both in the `doc` function and in the documentation, you'll see a `[source]` link.  This is like `??` but what the source link does is it takes you into the exact line of code in GitHub, so you can see exactly how that's implemented and what else is around it.  So, lots of good stuff there.

###  `59:10` Learning Rates
#### Questions from class 
- RT:  Why are you using 3's for your **learning rates** earlier?  With `3e-5` (3 e neg 5) and `3e-4` (3 e neg 4)?
- JH:  We found that `3e-3` is just a really good default learning rate.  It works most of the time for your initial fine-tuning before you unfreeze.  And then, I tend to kind of just multiply from there.  So, I generally find then that the next stage I will pick 10 times lower than that, so the second part of the slice and whatever the LR finder found for the first part of this slice.  The second part of the slice doesn't come from the LR finder.  It's just a rule of thumb which is like 10 times less than your first part which defaults to `3e-3`.  And then the first part of the slice is what comes out of the LR finder.  And, we'll be learning a lot more about these learning rate details, both today and in the coming lessons.  But, yeah, for now, all you need to remember is that in your, you know... your basic approach looked like this.  It was learn.fit one cycle, some number of epochs, I often pick 4 and some learning rate which defaults to `3e-3`.  I'll just type it out fully so you can see :
```python
learn.fit_one_cycle(4, 3e-3)
```
- And then we do that for a bit.  And then we unfreeze it:
```python
learn.unfreeze()
```
- And then we learn some more.  And so this is the bit where I just take whatever I did last time and **divide it by 10** (3e-3 ---> 3e-4) and then I also write like that [add slice], and then I have to put one more number in here `xxx`, and that's the number I get from the learning rate finder; the bit where it's got the strongest slope.  
```python
learn.fit_one_cycle(4, slice(xxx, 3e-4))
```
- So, that's kind of the... kind of don't have to think about, don't really have to know what's going on, rule-of-thumb that works, most of the time. 

### `1:01:10` Digging in to the math
go to lesson_2_2_lecture.md

