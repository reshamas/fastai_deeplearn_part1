

Please login as the user "ubuntu" rather than the user "root".

```bash
% pwd
/Users/reshamashaikh/.ssh
% 
ssh -i "id_rsa.pub" ubuntu@ec2-34-198-228-48.compute-1.amazonaws.com -L8888:localhost:8888
```

### `git pull` 
```bash
(fastai) ubuntu@ip-172-31-2-59:~$ ls
data  fastai  src
(fastai) ubuntu@ip-172-31-2-59:~$ cd fastai
(fastai) ubuntu@ip-172-31-2-59:~/fastai$ git pull
(fastai) ubuntu@ip-172-31-2-59:~/fastai$
```

### `conda env update`
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

#### get Jupyter Notebook sessions
```
jupyter notebook list
```

#### list CPU GPU memory usage:  
```
htop
```

### `scp` Secure Copy
```bash
% scp -r . ubuntu@107.22.140.44:~/data/camelhorse 
```



