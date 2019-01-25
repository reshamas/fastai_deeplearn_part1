# Lesson 2:  Intro to NN

#### What are two ways to go from an AND perceptron to an OR perceptron?  
- Increase the weights
- Decrease the magnitude of the bias

## Activation Functions
- Sigmoid:  
    - for large values, it is **1**
    - for small values, it is **0**
    - for 0, it is **0.5**
- Softmax:  
    - equivalent of **Sigmoid** activation function, but when the problem has 3 or more classes
    - **Scores** are computed from the **weights** and **inputs (or features)**
    - **Scores** are converted to **Probabilities**
    - results in probabilities that add to 1
- Tanh (hyperbolic tangent):
    - range of tanh(x) is:  (-1, 1)
    - this function is similar to sigmoid, but since the range is between -1 and 1, the derivatives are larger
    - the small difference actually lead to great advances in neural networks
- ReLU (rectified linear unit):
    - this is a very simple function
    - relu(x) = x, if x >= 0
    - relu(x) = 0, if x < 0
    - another way of seeing it is as the maximum between x and zero.
    - this function is used a lot instead of the sigmoid and it can improve the training significantly without sacrificing much accuracy since the derivative is 1 if the number is positive.  It's fascinating that this function which barely breaks linearity can lead to such complex non-linear solutions. 
    - this is the simplest non-linear function you can use
    - So, now with better activation functions, when we multiply derivatives to obtain the derivative to any sort of weight, the products will be made of slightly larger numbers which will make the derivative less small and will allow us to do gradient descent.    
    - turns out networks train much faster **with ReLU** than Sigmoid or Tanh (hyperbolic tangent)


## Lesson 2.10: Perceptron Algorithm
Reference:  https://pytorchfbchallenge.slack.com/archives/CE1TH8LPL/p1542288558155600?thread_ts=1542286943.143300  

```python
# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
   for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
```    

## Sigmoid Function
- for large values, it is **1**
- for small values, it is **0**
- for 0, it is **0.5**

## Softmax Function
- equivalent of **Sigmoid** activation function, but when the problem has 3 or more classes
- Notice that the probabilities need to add to 1.
- **Scores** are computed from the **weights** and **inputs (or features)**
- How do we turn **Scores** into **Probabilities**?
    - P(duck) 0.67:  2
    - P(beaver) = 0.24 : 1
    - P(walrus) = 0.09 :  0
- How can we turn this into an idea that works if we have scores that are negative numbers?
    - it's almost like we need to turn these scores into positive scores
    - How do we do this?  is there a function that can help us?
- Let's take each score, and divide it by the sum of each score.
    - duck:  2 / (2 + 1 + 0) = 2/3 = 0.67
    - beaver: 1 / (2 + 1 + 0) = 1/3 = 0.33
    - walrus: 0 / (2 + 1 + 0) = 0/3 = 0.00
    - X:  doesn't work, because we may have a score that is a negative number:  1 / (1 + 0 + -1) = 1/0 !
- Let's look at some options:
    - sin, cos, log, exponential
- What function turns every number into a positive number:  **exp**
- So, let's do exponential here
    - duck:   e^2 / (e^2 + e^1 + e^0) = 0.67
    - beaver: e^1 / (e^2 + e^1 + e^0) = 0.24
    - walrus: e^0 / (e^2 + e^1 + e^0) = 0.09

## Likelihood
- in summary, we really want to stay away from **products**
- We need to find a function that will help us turn products into sums 
- **logarithm!**
- log(ab) = log(a) + log(b)

## Cross-entropy
- **the sum of negatives of logarithms of probabilities**
- a good model will give us a low cross-entropy
- a bad model will give us a high cross-entropy
- we can think of the negatives of these logarithms as errors at each point.  
    - points that are correctly classified will have small errors
    - points that are **mis-classified** will have large errors

### Cross-entropy really says the following:
- If I have a bunch of events, and a bunch of probabilities, how likely is it that those events happen based on the probabilities?
    - if it's **very likely** then we have a **small cross-entropy**
    - if it's **unlikely** then we have a **large cross-entropy**
- cross entropy tells us when two vectors are similar or different
- 
### Cross-entropy:  takeaway
- **Yes, cross-entropy is inversely proportional to the total probability of an outcome.**

## Gradient
- If a point is well classified, we will get a small gradient. And if it's poorly classified, the gradient will be quite large.
- The gradient of the Error Function is the **partial derivative of E, with respect to each weight**

### Notebook
http://localhost:8888/notebooks/deep-learning-v2-pytorch/intro-neural-networks/gradient-descent/GradientDescent.ipynb

## Perceptron vs Gradient Descent
### Gradient Descent
- Change w_i to:  w_i + alpha * (y-y_hat) * x_i

### Perceptron
- If x is misclassified:
    - Change w_i to w_i **+** alpha * x_i, if **positive**
    - Change w_i to w_i **-** alpha * x_i, if **negative**
- If correctly classified:
    - y - y_hat = 0
- If misclassified:
    - y - y_hat = 1, if positive
    - y - y_hat = -1, if negative  
    
### Backpropagation

## Scaling the data
The next step is to scale the data. We notice that the range for grades is 1.0-4.0, whereas the range for test scores is roughly 200-800, which is much larger. This means our data is skewed, and that makes it hard for a neural network to handle. Let's fit our two features into a range of 0-1, by dividing the grades by 4.0, and the test score by 800.

### MinMaxScaler
The MinMaxScaler is the probably the most famous scaling algorithm, and follows the following formula for each feature:

xi–min(x) / max(x)–min(x)
It essentially shrinks the range such that the range is now between 0 and 1 (or -1 to 1 if there are negative values).

This scaler works better for cases in which the standard scaler might not work so well. If the distribution is not Gaussian or the standard deviation is very small, the min-max scaler works better.

However, it is sensitive to outliers, so if there are outliers in the data, you might want to consider the Robust Scaler below.

## Regularization
- large coefficients ---> overfitting
- penalize large weights:  (w_1, w_2, ...., w_n)
- add to Error Function
    - L1 regularization: lambda*(add sums of absolute values of weights)
    - L2 regularization:  lambda*(sum of weights squared)
- lambda is for how much we want to penalize the coefficients
    - if lambda is large, we penalize a lot
    - if lambda is small, then we don't penalize much
    
### L1 Regularization
- we end up sparse vectors
- sparsity:  (1, 0, 0, 1, 0)
- good for feature selection; small weights will go to 0.

### L2 Regularization
- tends not to favor sparse vectors
- sparsity:  (0.5, 0.3, ... , -0.2, 0.4, 0.1)
- normally better for training models

## Dropout
- if prob = 0.20, then the probability that each node will be dropped is 0.20

## Gradient Descent
### Local Minima
- we may hit a minimum for a while, but it may not be a global minimum.

### Random Restart
- start from a few different, random places and do gradient descent from all of them
- this increases the probability we will get to the global minimum, or at least a pretty good local minimum.

### Vanishing Gradient
- when the derivative gets close to zero, that is not good, since the derivative tells us which direction to move
- this gets even worse in most linear perceptrons
- the best way to fix this is to change the activation function

### hyperbolic tangent function  
#### tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- range of tanh(x) is:  (-1, 1)
- this function is similar to sigmoid, but since the range is between -1 and 1, the derivatives are larger
- the small difference actually lead to great advances in neural networks

### ReLU
- another very popular activation function is the Rectified Linear Unit (ReLU)
- this is a very simple function
- relu(x) = x, if x >= 0
- relu(x) = 0, if x < 0
- another way of seeing it is as the maximum between x and zero.
- this function is used a lot instead of the sigmoid and it can improve the training significantly without sacrificing much accuracy since the derivative is 1 if the number is positive.  It's fascinating that this function which barely breaks linearity can lead to such complex non-linear solutions. 
- So, now with better activation functions, when we multiply derivatives to obtain the derivative to any sort of weight, the products will be made of slightly larger numbers which will make the derivative less small and will allow us to do gradient descent.  

### ReLU
- this is the simplest non-linear function you can use
- turns out networks train much faster with ReLU than Sigmoid or Tanh (hyperbolic tangent)

### Multi-layer Perceptron
- note that even when using ReLU in multi-layers, the last layer must be sigmoid, since our final output still needs to be a probability between 0 and 1
- however, if we let the final unit be a ReLU, we can actually end up with regression models, that predict a value
- this will be of use in the recurring neural network section of the nanodegree

## Batch vs Stochastic Gradient Descent
- if the data is well-distributed, it's almost as if a small subset of it would give us a pretty good idea of what the gradient would be
- maybe it's not the best estimate for the gradient, but it is quick, and since we are iterating, it may be a good idea
- we take small subsets of data, run them through the neural network, calculate the gradient of the error function based on those points and then move one step in that direction.  
