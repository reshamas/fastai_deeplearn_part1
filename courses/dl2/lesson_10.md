# Lesson 10:  NLP Classification and Translation
(02-Apr-2018, live)  

### Part 2 / Lesson 10
- [Wiki Lesson 10](http://forums.fast.ai/t/part-2-lesson-10-wiki/14364)
- [Video Lesson 10](https://www.youtube.com/watch?v=h5Tz7gZT9Fo&feature=youtu.be) 
  - video length:  2:07:55
- http://course.fast.ai/lessons/lesson10.html
- Notebook:  
   * [pascal.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/pascal.ipynb)

---

## Review of Last Week
- if you are finding the material difficult, that's ok
- there are quite a few in-class fastai students who are doing this full-time and some of them are struggling with this material
- one of the reasons Jeremy had that content early on is we've got something to **cogitate** and think about and gradually work towards so by Lesson 14, you'll get a second crack at it

## Reading Papers (Research Journal Publications)
- not enough students are reading the papers!  The papers are the real ground truth
- a lot of people aren't reading the papers because they think they can't read the papers, but YOU ARE, because you are here

### Debugger
```python
pdb.set_trace()
```
### `00:14:20` NLP
- we've seen the idea of taking a pre-trained model, whip off some stuff from the top, replace it with new stuff, get it to do something similar
- we've dived a little bit deeper with that; with ConvLearner.pretrained, it had a standard way of sticking stuff on the top which does a particular thing, which was classification
- and then we learned, we can stick any PyTorch module at the end and have it do anything we like with a custom head
- suddenly you discover there are some interesting things we can do
- `00:15:35` YangLu said, what if we did a different kind of custom head? 
    - take original picture, rotate it, and then make our dependent variable the opposite of that rotation
    - and see if it can learn to un-rotate it
    - and this is a super useful thing, Google photos has an option to automatically rotate photos for you
    - as Yang Lu showed here, you could build that network right now by doing exactly the same as our previous lesson, but your custom head is one that spits out a single number, which is how much to rotate by and your dataset has a dependent variable which is how much did you rotate by?  And your dataset has a dependent variable, which is how much did you rotate by?
- so, you suddenly realize the idea of a backbone with a custom head, you can do almost anything you can think about
- Next, let's think about, how does that apply to NLP?


### Match the Equations to the Code
- [List of Mathematical Symbols](https://en.wikipedia.org/wiki/List_of_mathematical_symbols)

### Re-create things (graphs) you see in the papers
- "Try to recreate this chart, and make sure you understand what it's saying and why it matters"

### `00:16:40` We've moved from *torchtext* to *fastai.text*
#### NLP
- in the next lesson, we're going to go further and say, if NLP and computer vision lets you do the same basic ideas, then how do we combine the two? and we're going to learn about a model that we'll learn to find word structures from images, OR images from word structures, OR images from images!
- and that will form the basis, if you wanted to go further, of doing things from an image to a sentence, "image captioning", or going from a sentence to an image, which we will start to do "phrased image"
- so, from there, we'll go deeper into computer vision to think about what other kinds of things we can do with this idea of pre-trained network + custom head
- `00:17:45` we'll look at various kinds of image enhancement, like:
  - increasing the resolution of a low-res photo to guess what was missing
  - or adding artistic filters on top of photos
  - or changing photos of horses into photos of zebras and stuff like that 
- and finally, that will bring us all the way back to bounding boxes again
- to get there, we will first learn about segmentation, which is not just figuring out where the bounding box is, but figuring out what every single pixel in an image is part of
  - is this pixel part of a person, a car? 
- and then we will use that idea, idea of **unet** and it turns out this idea from unet, we can apply the idea of bounding boxes, which are called **feature pyramids** (everything has to have a different name in every slightly different area
- 

#### 4 Reasons Why..  *torchtext* to *fastai.text*
- no parallel processing
- hard to do simple things (like multi-label classification)
- no obvious way to save intermediate calculations
- somewhat convoluted API


- looking at transformations to images and asking "how does that apply to NLP?"
- in Part 1 we used the library, torchtext, but JH has found problems with it that are limiting; it's very slow because it's not doing parallel processing and doesn't remember what it did previously, so reruns it
- refresher: look at lesson 4, IMDb reviews
- we take sentences and turn them into numbers
- 





