# Takeways / Tips

## Modeling 
1.  When training a model, we can "ignore" or not worry as much about **overfitting** as long as the validation error is decreasing.


2.  **Image Sizes** are generally at 224x224 and 299x299, which are the sizes that imagenet models are generally trained at. You get best results if you use the same as the original training size. Since people don’t tend to mention what size was used originally, you can try using both with something like dogs v cats and see which works better. More recent models seem to generally use 299.

3.  **Rare Cases**  You can replicate the rare classes to make them more balanced. Never throw away data!

### Reducing Overfitting
* data augmentation
* pretrained network
* gradually increasing image size
* differential learning rates
* SGDR
* dropouts
* higher resolution images

# Best Practices

1.  When opening a notebook in fastai library, make a copy with the prefix **tmp**.  "tmp" files are included in fastai repo's [.gitignore](https://github.com/fastai/fastai/blob/master/.gitignore)

# Solving Errors
1.  Do `git pull` of [fastai library](https://github.com/fastai/fastai).  Updates may sort out some errors.
```bash
git pull
conda env update
conda update --all 
```
2.  Delete `tmp` directory and rerun  

3.  CUDA out of memory error:  
    - interrupt kernel
    - reduce batch size
    - **RESTART kernel**!

# Debugging
Note from Jeremy:  
Immediately after you get the error, type %debug in a cell to enter the debugger. Then use the standard python debugger commands to follow your code to see what’s happening. 

