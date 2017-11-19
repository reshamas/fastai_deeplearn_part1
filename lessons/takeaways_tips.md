# Takeways / Tips

1.  When training a model, we can "ignore" or not worry as much about **overfitting** as long as the validation error is decreasing.


2.  **Image Sizes** are generally at 224x224 and 299x299, which are the sizes that imagenet models are generally trained at. You get best results if you use the same as the original training size. Since people don’t tend to mention what size was used originally, you can try using both with something like dogs v cats and see which works better. More recent models seem to generally use 299.

## Tips

### Errors
1.  Delete `tmp` directory an rerun
