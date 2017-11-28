# fastai & Deep Learning & Other Terms

---
# fastai Terms
`bs` = batch size  
`sz` = size (of photo)  
`tfms` = transformations  
`wds` = weight decays  

---
# Deep Learning Terms

### SoTA (State-of-the-Art)

### TTA (Test Time Augmentation)

### Epoch
An epoch is a complete pass through a given dataset.

### ADAM
- Adam is a stochastic gradient descent algorithm based on estimation of 1st and 2nd-order moments. The algorithm estimates 1st-order moment (the gradient mean) and 2nd-order moment (element-wise squared gradient) of the gradient using exponential moving average, and corrects its bias. The final weight update is proportional to learning rate times 1st-order moment divided by the square root of 2nd-order moment.
- Adam takes 3 hyperparameters: the learning rate, the decay rate of 1st-order moment, and the decay rate of 2nd-order moment
- [ADAM: A Method for Stochastic Optimization](https://theberkeleyview.wordpress.com/2015/11/19/berkeleyview-for-adam-a-method-for-stochastic-optimization/)

### Stochastic gradient descent (SGD)
Stochastic gradient descent (often shortened to SGD), also known as incremental gradient descent, is a stochastic approximation of the gradient descent optimization and iterative method for minimizing an objective function that is written as a sum of differentiable functions. In other words, SGD tries to find minima or maxima by iteration.

---
# Unix
scp :  secure copy  

---



### References
- [Full Ranking List](https://github.com/thedataincubator/data-science-blogs/blob/master/output/DL_libraries_final_Rankings.csv)
- [Ranking Popular Deep Learning Libraries for Data Science](https://blog.thedataincubator.com/2017/10/ranking-popular-deep-learning-libraries-for-data-science/) Oct 2017




---
# NLP Terms
```
POS = part of speech  
NP-chunking = noun phrase chunking  

DT = determiner
JJ = adjectives
NN = noun
VBD = verb
```

### IE (Information Extraction)
IE turns the unstructured information embedded in texts into structured data. 


### IOB (Inside, Outside, Beginning)
```
The most widespread file representation uses IOB tags:
IOB = Inside-Outside-Begininning
B = beginnning (marks beginning of chunk)
I = inside (all subsequent parts of chunk)
O = outside
```

### Named Entity
anything that can be referred to with a proper name


### NER (Named Entity Recognition)
task of detecting and classifying all the proper names mentioned in a text  
* Generic NER:  finds names of people, places and organizations that are mentioned in ordinary news texts
* practical applications:  built to detect everything  from names of genes and proteins, to names of college courses

### Reference Resolution (Coreference)
occurs when two or more expressions in a text refer to the same person or thing; they have the same referent, e.g. Bill said he would come; the proper noun Bill and the pronoun he refer to the same person, namely to Bill

### Relation Detection and Classification
find and classify semantic relations among the entities discovered in a given text

### Event Detection and Classification
find and classify the events in which the entities are participating

### Temporal Expression Detection
* tells us that our sample text contains the temporal expressions *Friday* and *Thursday*
* includes date expressions such as days of the week, months, holidays, as well as relative expressions including phrases like *two days from now* or *next year*.
* includes time:  noon, 3pm, etc.

### Temporal Analysis
over problem is to map temporal expressions onto specific calendar dates or times of day and then to use those times to situate events in time.  

---

# Deep Learning Terms

### ANN  (Artificial Neural Network)

### CNN / ConvNet (Convolutional Neural Network)
is a type of feed-forward artificial neural network in which the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex

### CUDA
CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia. ... When it was first introduced by Nvidia, the name CUDA was an acronym for Compute Unified Device Architecture, but Nvidia subsequently dropped the use of the acronym.

### GloVe
Global vectors

### RNN  (Recurrent Neural Network)
a class of artificial neural network where connections between units form a directed cycle. This creates an internal state of the network which allows it to exhibit dynamic temporal behavior. Unlike feedforward neural networks, RNNs can use their internal memory to process arbitrary sequences of inputs. 

### Dropout
* Regularization technique in neural networks to prevent overfitting
* dropping some neurons in a layer

### Reinforcement Learning


### ReLU
Rectified Linear Unit (ReLU) activation function, which is zero when x &lt 0 and then linear with slope 1 when x &gt 0. 

### LSTM  (Long Short-Term Memory-Networks)
* An LSTM unit is a recurrent network unit that excels at remembering values for either long or short durations of time. The key to this ability is that it uses no activation function within its recurrent components. Thus, the stored value is not iteratively squashed over time, and the gradient or blame term does not tend to vanish when Backpropagation through time is applied to train it.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
is a (prize-winning) technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. The technique can be implemented via Barnes-Hut approximations, allowing it to be applied on large real-world datasets. We applied it on data sets with up to 30 million examples. 


### BLEU (bilingual evaluation understudy) 
is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" – this is the central idea behind BLEU.[1][2] BLEU was one of the first metrics to achieve a high correlation with human judgements of quality,[3][4] and remains one of the most popular automated and inexpensive metrics.


