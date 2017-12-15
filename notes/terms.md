# fastai & Deep Learning & Other Terms

---
# fastai Terms
`bptt` back propagation through time  
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

### ANN  (Artificial Neural Network)

### CNN / ConvNet (Convolutional Neural Network)
is a type of feed-forward artificial neural network in which the connectivity pattern between its neurons is inspired by the organization of the animal visual cortex

### CUDA
CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia. ... When it was first introduced by Nvidia, the name CUDA was an acronym for Compute Unified Device Architecture, but Nvidia subsequently dropped the use of the acronym.

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

