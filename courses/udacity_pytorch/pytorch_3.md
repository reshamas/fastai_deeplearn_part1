# PyTorch

- `weights.reshape(a, b)`  will return a new tensor with the same data as `weights` with size `(a, b)` sometimes, and sometimes a clone, as in it copies the data to another part of memory 
- `weights.resize_` underscore at end means this method is an in-place operation

## Universal Function Approximator
- 

## Loss Function (Cost Function)
- it is a measure of our prediction error
- the whole goal is to adjust our network parameters to minimize our loss
- we do this by using a process called **gradient descent**

## Gradient
- the gradient is the slope of the loss function with respect to our parameters
- the gradient always points in the direction of the *fastest change*
- so, if we have a mountain, the gradient is always going to point up the mountain
- so, you can imagine our loss function being like this mountain where we have a high loss up here, and we have a low loss down here
- so, we know that we want to get to the minimum of our loss when we minimize our loss, and so we want to go downwards
- and so, basically, the gradient points upwards and so, we just go the opposite direction. So, we go in the direction of the negative gradient
- and then, if we keep following this down, then eventually we get to the bottom of this mountain, the **lowest loss**
- with multi-layered neural networks, we use an algorithm called backpropagation to do this

## Backpropagation
- backprop is really just an application of the chain rule of calculus
- So, if you think about it, when we pass in some data, some input into our network, it goes through this forward pass through the network to calculate our loss
- So, we pass in some data, some feature input x
  - and then it goes through this linear transformation which depends on our weights and biases.  
  - And then through some activation function like sigmoid
  - through another linear transformation with some more weights and biases
  - and then that goes in [last layer], and from that we calculate our loss
- So, if we make a small change in our weights (say in the first layer), it's going to propagate through the network and end up, like results in, a small change in our loss.
- So you can kind of think of this as a chain of changes
- So, with backprop, we actually use these same changes, but we go in the opposite direction
- So, for each of these operations like the loss and the linear transformation (L2), and the sigmoid activation function, there's always going to be some derivative, some gradient between the outputs and the inputs
- And so what we do, is we take each of the gradients for these operations and we pass them backwards through the network.  
- At each step, we multiply the incoming gradient with the gradient of the operation itself.  
- So, for example, just kind of starting at the end with the loss 
- so we pass this gradient through the loss, dl/dL2, so this is the gradient of the loss with respect to the second linear transformation
- and then we pass that backwards again and if we multiply it by the loss of this L2, so this is the linear transformation with respect to the outputs of our activation function, that gives us the gradient for this operation
- And if you multiply this gradient by the gradient coming from the loss, then we get the total gradient for both of these parts 
- and this gradient can be passed back to this softmax function
- So, as the general process for backpropagation, we take our gradients, we pass it backwards to the previous operation, multiply it by the gradient there, and then pass that total gradient backwards.
- So, we just keep doing that through each of the operations in our network. 

## Losses in PyTorch
- PyTorch provide a lot of losses, including the cross entropy loss
- `criterion = nn.CrossEntropyLoss`
- Cross entropy loss is used in classification problems
- So, if we wanted to use cross-entropy, we just say `criterion = nn.crossEntropyLoss` and create that class
- So, one thing to note, if you look at the documentation for cross-entropy loss, you'll see that it actually wants the scores, like the logits, of our network, as the input to the cross-entropy loss.  
- So, you'll be using this with an output such as softmax, which gives us this nice probability distribution.  But, for computational reasons, then it's generally better to use the logits which are the input to the softmax as the input to this loss.
- So, the input is expected to be the scores for each class, and not the probabilities themselves.  
- So, first I am going to import the necessary modules.  

## Metrics
- Accuracy
- Precision
- Recall
- Top-5 Error Rate
- `ps.topk(1)` returns the highest value (or probability) for a class

## Transfer Learning
- Using a pre-trained network on images not in the training set is called transfer learning. 
- Most of the pretrained models require the input to be 224x224 images.






