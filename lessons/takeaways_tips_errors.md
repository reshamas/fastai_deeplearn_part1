# Takeways / Tips

1.  When training a model, we can "ignore" or not worry as much about **overfitting** as long as the validation error is decreasing.


2.  **Image Sizes** are generally at 224x224 and 299x299, which are the sizes that imagenet models are generally trained at. You get best results if you use the same as the original training size. Since people don’t tend to mention what size was used originally, you can try using both with something like dogs v cats and see which works better. More recent models seem to generally use 299.

3.  **Rare Cases**  You can replicate the rare classes to make them more balanced. Never throw away data!

# Solving Errors
1.  Do `git pull` of [fastai library](https://github.com/fastai/fastai).  Updates may sort out some errors.
2.  Delete `tmp` directory and rerun
3.  CUDA out of memory error:  
    - interrupt kernel
    - reduce batch size
    - **RESTART kernel**!
