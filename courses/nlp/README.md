# fastai:  [NLP Course](https://www.fast.ai/2019/07/08/fastai-nlp/)
- [YouTube Playlist](https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9)
- [Jupyter notebooks on GitHub](https://github.com/fastai/course-nlp)

## Lectures 

1. [What is NLP?](www.youtube.com/watch?v=cce8ntxP_XI) (0:23)  (done 03-Mar-2020)
2. [Topic Modeling with SVD & NMF (NLP video 2)](www.youtube.com/watch?v=tG3pUwmGjsc) (1:07)  (done 03-Mar-2020)
3. [Topic Modeling and SVD revisited](https://youtu.be/lRZ4aMaXPBI) (33:06)
4.  (58:20)
5. 
...
19.


## Lesson 2 [Topic Modeling with SVD & NMF (NLP video 2)](www.youtube.com/watch?v=tG3pUwmGjsc)
* spacy doesn't offer a stemmer, because it doesn't think it should be used
* Google [sentencepiece](https://github.com/google/sentencepiece)
  * performs sub-word tokens
* NMF (non-negative matrix factorization) is not unique, but can be more interpretable

To check time of a step:  
```python
%time u, s, v = np.linalg.svd(vectors, full_matrices=False)
```

## Lesson 3

- stemming:  getting roots of words  (chops off end, "poor man's lemmatization")
- lemmatization:  (fancier)
- lemmatization is more computationally expensive than stemming
- stemming is quicker and easier
 
### Pre-processing
- when you have less data, do this pre-processing
- do you think your model can handle the complexity:  
 - if you're using neural networks, don't do lemmatization, because that is throwing away information
 - if you have a simpler model, can't learn as much complexity, so do this pre-processing

### Factorization is analagous to matrix decomposition

### What are the nice properties that matrices in an SVD decomposition have?
- A = USV
- U:  orthonormal; columns or rows are orthonormal to each other; the columns are orthogonal and pairwise normalized.  (dot product of two columns is 0. dot product of column with itself gives us 1)
- S:  diagonal matrix; everything off diagonals is 0; capture an idea of importance, singular values, descending order: capture biggest one first, non-negative, scale of U and V is both 1
- V:  same properties as U, but transpose of rows are orthonormal to each other
- NMF:  special property in decomposition is **non-negative** AND matrix is **sparse** (sparse means many of the values are zero)




