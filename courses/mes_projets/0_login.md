# Logging in to AWS

## Step 0:  Initial Set-up Assumptions
Assuming:  
- I have launched a p2 instance
- I have set up my key pair
- I have created an alias in my startup configuration file  `~/.bash_profile`.  In my case, I use `~/.zshrc`

My alias:  
```bash
alias fastai='ssh -i "id_rsa" ubuntu@ec2-88-888-888-88.compute-1.amazonaws.com -L8888:localhost:8888'
```

## Step 1:  AWS Console
- sign in here:  console.aws.amazon.com/
- start my `p2.xlarge` instance from before

## Step 2:  My terminal on my Mac (local computer)

### Go to the appropriate directory
```bash
cd /Users/reshamashaikh/.ssh
```
### Login to AWS
Login as the user "ubuntu" rather than the user "root".

```bash
fastai
```

### Update Ubuntu: `sudo apt-get update`
```bash
sudo apt-get update
```

### Update fastai repo:  `git pull` 
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
### Update Anaconda packages:  `conda env update`
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
## Update Anaconda packages:  `conda update --all`


## Step 3:  Turn off AWS Instance after completing work!

---
# My Projects

## my code  
```bash
(fastai) ubuntu@ip-172-31-2-59:~/git_repos/projects$ pwd 
/home/ubuntu/git_repos/projects
(fastai) ubuntu@ip-172-31-2-59:~/git_repos/projects$ ls -l
total 12
drwxrwxr-x 2 ubuntu ubuntu 4096 Jan  8 21:07 camels_h
drwxrwxr-x 3 ubuntu ubuntu 4096 Jan  8 00:44 iceberg
-rw-rw-r-- 1 ubuntu ubuntu   23 Jan  7 21:04 README.md
(fastai) ubuntu@ip-172-31-2-59:~/git_repos/projects$ 
```

## my data
```bash
(fastai) ubuntu@ip-172-31-2-59:~/data$ pwd
/home/ubuntu/data
(fastai) ubuntu@ip-172-31-2-59:~/data$ ls -alt
total 20
drwxr-xr-x 20 ubuntu ubuntu 4096 Jan  8 21:11 ..
drwxrwxr-x  2 ubuntu ubuntu 4096 Jan  7 20:44 iceberg
drwxrwxr-x  5 ubuntu ubuntu 4096 Jan  7 20:38 .
drwxrwxr-x  8 ubuntu ubuntu 4096 Dec 21 01:53 camelhorse
drwxrwxr-x  8 ubuntu ubuntu 4096 Dec 20 22:19 dogscats
(fastai) ubuntu@ip-172-31-2-59:~/data$ 
```

## launch Jupyter Notebook
```bash
(fastai) ubuntu@ip-172-31-2-59:~$ pwd
/home/ubuntu
(fastai) ubuntu@ip-172-31-2-59:~$ jupyter notebook
```

