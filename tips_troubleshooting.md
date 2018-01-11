# Solving Errors

1.  Do `git pull` of [fastai library](https://github.com/fastai/fastai).  Updates may sort out some errors.
```bash
git pull
```

2.  Update Anaconda packages
```bash
conda env update
conda update --all 
```

3.  Delete `tmp` directory and rerun  

4.  CUDA out of memory error:  
    - interrupt kernel
    - reduce batch size
    - **RESTART kernel**!

# Debugging
Note from Jeremy:  
Immediately after you get the error, type %debug in a cell to enter the debugger. Then use the standard python debugger commands to follow your code to see what’s happening. 
