# Lesson 1a:  Intro

(30-Oct-2017, live)  


### Wiki
[Wiki: Lesson 1](http://forums.fast.ai/t/wiki-lesson-1/7011)  

--- 

## USF
In-person Info:  [Deep Learning Certificate Part I](https://www.usfca.edu/data-institute/certificates/deep-learning-part-one)  

### Staff
* Intro by [David Uminsky](https://www.usfca.edu/faculty/david-uminsky), Director of Data Institute of USF 
* [Yannet Interian](https://www.usfca.edu/faculty/yannet-interian), Assistant Professor USF
* [Rachel Thomas](https://www.usfca.edu/data-institute/about-us/researchers), Researcher in Residence
* [Jeremy Howard](https://www.usfca.edu/data-institute/about-us/researchers), Distinguished Scholar in Deep Learning

### Classroom
* 200 students in room at USF
* 100 Masters' students upstairs at USF
* 400 International Fellows, via livestream

Being recorded and will become a fastai MOOC.  

#### Teams
* teams of 6 people
* get help with stuff

## Python
* using Python 3.6

## Platforms
* [Crestle](https://www.crestle.com) built by [Anurag Goel](https://www.linkedin.com/in/anuragoel/)  
* [Paperspace](https://www.paperspace.com)
* [AWS](https://aws.amazon.com/console/)

## Deep Learning
* Deep learning is a particular way of doing machine learning
* [Arthur Samuels](https://en.wikipedia.org/wiki/Arthur_Samuel)
  * he invented machine learning
  * rather than programming, step-by-step, give the computer *examples*
    * **let the computer figure out the problem by giving it examples**
  * let computer play checkers against itself thousands of times; it figured out which parameters worked the best
  * Samuel **Checkers-playing** Program appears to be the world's first self-learning program, and as such a very early demonstration of the fundamental concept of artificial intelligence (AI); 1962
  * he worked at Bell Labs and IBM, then Stanford Univ
  
### Machine Learning
#### Example:  ML Algorithm in Predicting Breast Cancer Survival Based on Pathology Slides
* start with pictures of breast cancer slides
* work with computer scientists, pathologists worked together to determine features that would predict who would survive or not, based on slides
* process of building model can take some time (many years); can pass data into ML algorithm, such as logistic regression; regression can determine which sets of features separate out the 2 classes
* this can work well, but requires a lot of experts and requires the feature data
* this ML algorithm was more accurate at predicting breast cancer survival than human pathologists 

#### Examples of ML Uses, Thanks to Deep Learning
* gmail, generates automatic responses to emails.
* skype, translate to different languages, in real time
* At Google, every single part of the company uses deep learning
* [DeepMind AI Reduces Google Data Centre Cooling Bill by 40%](https://deepmind.com/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-40/)
* [Baidu’s Deep-Learning System is better at English and Mandarin Speech Recognition than most people](https://www.nextbigfuture.com/2015/12/baidus-deep-learning-system-is-better.html)
* [How Google's AlphaGo Beat a Go World Champion](https://www.theatlantic.com/technology/archive/2016/03/the-invisible-opponent/475611/)
* [Splunk and Tensorflow for Security: Catching the...](https://www.splunk.com/blog/2017/04/18/deep-learning-with-splunk-and-tensorflow-for-security-catching-the-fraudster-in-neural-networks-with-behavioral-biometrics.html)
  * took mouse movements on web pages, turned them into pictures, tracking where mouse moved.  
  * fed data into convolutional neural network
  * used for fraud detection

#### Future Work
How do we get computers and humans to work better together?   

#### Societal Implications
* [The wonderful and terrifying implications of computers that can learn](https://www.ted.com/talks/jeremy_howard_the_wonderful_and_terrifying_implications_of_computers_that_can_learn) (Ted Talk by Jeremy Howard 2014)
* ML / DL algorithms need to be in the hands of practioners who understand the economics / implications of the algorithms.  
* practioners who understand societal implications; what kind of problems should be solved; what does a good solution look like...

#### Jeremy's Work
* [Enlitic](https://www.enlitic.com)
* [Exponential Medicine: Deep Learning AI Better Than Your Doctor at Finding Cancer](https://singularityhub.com/2015/11/11/exponential-medicine-deep-learning-ai-better-than-your-doctor-at-finding-cancer/)


#### Goal of This Course
* that people from all different backgrounds will use deep learning to solve problems

## Deep Learning
* deep learning is a way of doing machine learning
* way of giving machine data (examples) and having it figure out the problem that is represented in those examples

## What We Are Looking For:  Something That Has 3 Properties
(3 Things that Give Us Modern Deep Learning)  
We are looking for a **mathematical function** that is *so flexible* that it can solve any given problem.  
1. Infinitely Flexible Functions
2. All-Purpose Parameter Fitting (way to train the parameters)
  * things can fit hundreds of millions of parameters
3. Fast and scalable

Example of limitation:  linear regression is limited by the fact it can only represent linear functions.  

Deep Learning has all 3 of above properties.  
* functional form:  neural network
* multiple layers allows more complex relationships
* parameters of neural network can be found using gradient descent

### Gradient Descent
* approach works well in practice; local minima are "equivalent" in practice
* different optimization techniques determine how quickly we can find the way down.

### Key discoveries thru Theoretical Side
* Very, very simple architectures of neural network and very, very simple methods of gradient descent work best in most situations.  
* We'll learn how every step works, using simple math.  

### Fast and Scalable:  Made Possible by GPUs
* GPU = Graphical Processing Unit
* GPUs are used in video games
* Huge industry of video games accidentally built for us what we need to do deep learning
* GPUs are useful and needed for deep learning
* GPUs are 10x faster than CPUs
* Best hardware for deep learning:  NVIDIA GTX-1080 Ti for ~ $600

## Art of Learning
* [A Mathematician's Lament](https://www.maa.org/external_archive/devlin/LockhartsLament.pdf) by Paul Lockhart (25 pages)
* [40 Years of Teaching Thinking: Revolution, Evolution, and What Next?](https://www.youtube.com/watch?v=-nmt1atA6ag) video, 2011 (1 hr 12 min)

## Projects Done
* [How HBO’s Silicon Valley built “Not Hotdog” with mobile TensorFlow, Keras & React Native](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3) by Tim Anglade

## Work
* will need to put in 10 hours a week (in addition to lecture time)
* spend time **RUNNING THE CODE** (rather than researching the theory)
* create blog posts 

## The Test of Whether You Can Understand
* Deep Learning is about solving problems
  * if you can't turned it into code, you can't solve the problem.  
* You can code / build something with it
* You can explain / teach it to someone else
  * Write a blog post
  * Help others who have questions
  
## Portfolio
* people are hired based on their portfolio (not USF DL certificate)
* GitHub projects, blog posts --> **can get hired based on portfolio**
* write down what you are learning in a form that other people can understand

## Goal
* main goal is not to help you move to a deep learning job
* continue doing what you're doing and bring deep learning to that
* examples:  medicine, journalism, dairy farming
* opportunities to change society
* focus:  help you be a great practitioner of deep learning
* opportunity - doing things differently
* come up with a project idea

