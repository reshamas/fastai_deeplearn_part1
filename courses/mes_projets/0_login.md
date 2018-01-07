# logging in

## login
Please login as the user "ubuntu" rather than the user "root".

```bash
% pwd
/Users/reshamashaikh/.ssh
% 
ssh -i "id_rsa.pub" ubuntu@ec2-34-198-228-48.compute-1.amazonaws.com -L8888:localhost:8888
```

## update Ubuntu: `sudo apt-get update`
```bash
sudo apt-get update
```


## update fastai repo:  `git pull` 
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

## update Anaconda packages:  `conda env update`
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

### get list of Jupyter Notebook sessions
```
jupyter notebook list
```

### list CPU GPU memory usage:  
```
htop
```



