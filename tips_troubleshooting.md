# Solving Errors

## Latest version of fastai library
Do <kbd> git pull </kbd> of [fastai library](https://github.com/fastai/fastai).  Updates may sort out some errors.
```bash
git pull
```
## Update Anaconda packages
```bash
conda env update
conda update --all 
```

## Delete `tmp` directory and rerun  

## CUDA out of memory error
- interrupt kernel
- reduce batch size
- **RESTART kernel**!

## TTA (Test Time Augmentation)
- [forum post](http://forums.fast.ai/t/lesson-2-dog-breeds-error-on-call-of-accuracy-log-preds-y/11965)
- "TTA used to return the average of the augmentations as a prediction. Now it returns the set so you can do with them as you please."

#### Error with this code
```python
log_preds,y = learn.TTA()
probs = np.exp(log_preds)
accuracy(log_preds,y), metrics.log_loss(y, probs)
```
#### Adjust with this code
```python
log_preds,y = learn.TTA()
preds = np.mean(np.exp(log_preds),0)
```

---
## Empty graph with learning rate finder
- try increasing the batch size

---

# Debugging
Note from Jeremy:  
Immediately after you get the error, type `%debug` in a cell to enter the debugger. Then use the standard python debugger commands to follow your code to see what’s happening. 
