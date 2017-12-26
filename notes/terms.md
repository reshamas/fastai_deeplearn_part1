# fastai & Deep Learning & Other Terms

---
# fastai Terms
`bptt` back propagation through time  
`bs` = batch size  
`.cuda()` is we tell it manually to use the GPU  
`md.nt` = number of unique tokens  
`n_fac` = size of embedding  
`lo` = layer optimizer  
`sz` = size (of photo)  
`tfms` = transformations  
`wds` = weight decays  

---
# Other Terms
cardinality:  number of levels of a categorical variable

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


### SGDR (Stochastic Gradient Descent with Restart)
Restart techniques are common in gradient-free optimization to deal with multimodal
functions. Partial restarts are also gaining popularity in gradient-based optimization
to improve the rate of convergence in accelerated gradient schemes to
deal with ill-conditioned functions. In this paper, we propose a simple restart technique
for stochastic gradient descent to improve its anytime performance when
training deep neural networks. We empirically study its performance on CIFAR-
10 and CIFAR-100 datasets where we demonstrate new state-of-the-art results
below 4% and 19%, respectively. Our source code is available at
https://github.com/loshchil/SGDR.

https://pdfs.semanticscholar.org/7edd/785cf90e5a218022904585208a1585d634e1.pdf


### ANN  (Artificial Neural Network)

### CNN / ConvNet (Convolutional Neural Network)
is a type of feed-forward artificial neural network in which the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex

### CUDA
CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia. ... When it was first introduced by Nvidia, the name CUDA was an acronym for Compute Unified Device Architecture, but Nvidia subsequently dropped the use of the acronym.

### DNN (Deep Neural Networks)

### Dropout
* Regularization technique in neural networks to prevent overfitting
* dropping some neurons in a layer


### Gradient Boosting
Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.


### GRU (Gated Recurrent Unit)

### LSTM  (Long Short-Term Memory-Networks)
* An LSTM unit is a recurrent network unit that excels at remembering values for either long or short durations of time. The key to this ability is that it uses no activation function within its recurrent components. Thus, the stored value is not iteratively squashed over time, and the gradient or blame term does not tend to vanish when Backpropagation through time is applied to train it.


### QRNNs (Quasi-Recurrent Neural Networks) 
an approach to neural sequence modeling that alternates convolutional layers, which apply in parallel across timesteps, and a minimalist recurrent pooling function that applies in parallel across channels. Despite lacking trainable recurrent layers, stacked QRNNs have better predictive accuracy than stacked LSTMs of the same hidden size. Due to their increased parallelism, they are up to 16 times faster at train and test time. Experiments on language modeling, sentiment classification, and character-level neural machine translation demonstrate these advantages and underline the viability of QRNNs as a basic building block for a variety of sequence tasks.
https://arxiv.org/abs/1611.01576  


### RNN  (Recurrent Neural Network)
a class of artificial neural network where connections between units form a directed cycle. This creates an internal state of the network which allows it to exhibit dynamic temporal behavior. Unlike feedforward neural networks, RNNs can use their internal memory to process arbitrary sequences of inputs. 



### Reinforcement Learning


### ReLU (Rectified Linear Unit)
ReLU is an activation function, which is zero when x &lt 0 and then linear with slope 1 when x &gt 0. 


### t-SNE (t-Distributed Stochastic Neighbor Embedding)
is a (prize-winning) technique for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets. The technique can be implemented via Barnes-Hut approximations, allowing it to be applied on large real-world datasets. We applied it on data sets with up to 30 million examples. 

