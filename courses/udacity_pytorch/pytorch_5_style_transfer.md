# Style Transfer

## Gram Matrix
- non-localized information is information that would still be there even if the image was shuffled around in space
- style:  prominent colors and textures of an image
- gram matrix:  whose values indicate the similarities between the layers
  - dimensions don't depend on the input image
  - just one mathematical way of representing shared or prominent styles
- style itself is kind of an abstract idea.  but the gram matrix is the most widely used in practice

## Style Loss
- the smaller the alpha/beta ratio, the more stylistic effect you will see.
- alpha:  content weight
- beta:  style weight

## VGG Features

## Lesson 8 Notebook (Exercise)
- https://github.com/udacity/deep-learning-v2-pytorch/blob/master/style-transfer/Style_Transfer_Solution.ipynb

