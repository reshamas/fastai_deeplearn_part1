# Logging in to AWS

### login
Please login as the user "ubuntu" rather than the user "root".

```bash
% pwd
/Users/reshamashaikh/.ssh
```

Note:  create an alias in `.zshrc` to log into AWS computer
```bash
% 
fastai
```

### update Ubuntu: `sudo apt-get update`
```bash
sudo apt-get update
```

### update fastai repo:  `git pull` 
```bash
cd fastai
```
```bash
git pull
```

>my example
```bash
(fastai) ubuntu@ip-172-31-2-59:~$ ls
data  fastai  src
(fastai) ubuntu@ip-172-31-2-59:~$ cd fastai
(fastai) ubuntu@ip-172-31-2-59:~/fastai$ git pull
(fastai) ubuntu@ip-172-31-2-59:~/fastai$
```

### update Anaconda packages:  `conda env update`
```bash
conda env update
```
>my example
```bash
(fastai) ubuntu@ip-172-31-2-59:~/fastai$ conda env update
Using Anaconda API: https://api.anaconda.org
Fetching package metadata .................
Solving package specifications: .
#
# To activate this environment, use:
# > source activate fastai
#
# To deactivate an active environment, use:
# > source deactivate
#
(fastai) ubuntu@ip-172-31-2-59:~/fastai$
```
## `conda update --all`


---
# My Projects
I'm working in this directory  
```bash
(fastai) ubuntu@ip-172-31-2-59:~/git_repos/projects$ pwd
/home/ubuntu/git_repos/projects
```


---

### get list of Jupyter Notebook sessions
```
jupyter notebook list
```

### list CPU GPU memory usage:  
```
htop
```

file:  `.bashrc`  
```
export PYTHONPATH=$PYTHONPATH:~/fastai
```
https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d



