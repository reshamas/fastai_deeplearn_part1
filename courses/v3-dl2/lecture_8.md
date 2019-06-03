# Lesson 8

## From foundations: Matrix multiplication; Fully connected network forward and backward passes

### Broadcasting
- powerful tool for writing code in Python that runs at C speed
- with PyTorch, it will run at CUDA speed;  allows us to get rid of our for-loops
- 'broadcasting' a scalar to a tensor
```python
t = c.expand_as(m)
t.storage()
t.stride(), t.shape
```
- tensors that behave as higher rank things than they are actually stored as
- broadcasting functionality gives us C like speed without additional memory overhead
- `unsqueeze` adds an additional dimension
```python
c.unsqueeze(1)
```

### Einstein Summation Notation


