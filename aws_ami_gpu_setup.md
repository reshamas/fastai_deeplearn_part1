# AWS fastami GPU Image Setup
fastai.ai Part 1 v2  

AMI (Amazon Machine Image):  a template for how your computer is created

### Getting Started
Log into AWS Console:  http://console.aws.amazon.com/  
Select Service:  **EC2**  
Launch Instance

### Step 1:  Choose an Amazon Machine Image (AMI)
* Search Community AMIs [left menu]
* Search:  `fastai`
* Select this image (for region N. Virginia):  `fastai-part1v2-p2 - ami-c6ac1cbc`


### Step 2:  Choose an Instance Type
(Note:  it is the kind of computer we want to use.)  
* Filter by:  `GPU Compute`
* Select:  `p2.xlarge`   (this is the cheapeast, reasonably effective for deep learning type of instance available)
* Select: `Review and Launch` at bottom

#### Step 2b:  Select keypair
Note:  you have already created a keypair in the past.  Use one of those.  For more specific instructions, see section "Create a Keypair."

**And, voila! We have just created a new computer on AWS that we can log into :boom:**

### Connect:  log into our AWS computer
* Get the Public IP address of your AWS computer
* `ssh` in to instance from your local computer.  Ensure you are in your `.ssh` directory.  
>my current path
```bash
pwd
/Users/reshamashaikh/.ssh
```

#### Syntax for logging in and setting up tunnel for Jupyter Notebook
Note 1:  you will put **your Public IP address** where mine is.  
Note 2:  This part `-L8888:localhost:8888` connects Jupyter Notebook from AWS to your computer.  
```
ssh -i aws_fastai_gpu.pem ubuntu@54.175.101.64 -L8888:localhost:8888  
```


### 
```bash
(fastai) ubuntu@ip-172-31-10-243:~$ ls
data  fastai  src
(fastai) ubuntu@ip-172-31-10-243:~$
```
```
(fastai) ubuntu@ip-172-31-10-243:~$ cd fastai
(fastai) ubuntu@ip-172-31-10-243:~/fastai$ ls -alt
total 76
drwxr-xr-x 18 ubuntu ubuntu  4096 Nov  7 16:25 ..
drwxrwxr-x  8 ubuntu ubuntu  4096 Nov  5 00:35 .git
drwxrwxr-x  6 ubuntu ubuntu  4096 Nov  5 00:35 fastai
drwxrwxr-x  6 ubuntu ubuntu  4096 Nov  5 00:29 .
-rw-rw-r--  1 ubuntu ubuntu  1273 Nov  5 00:29 environment.yml
drwxrwxr-x  3 ubuntu ubuntu  4096 Nov  1 21:30 tutorials
-rw-rw-r--  1 ubuntu ubuntu   905 Nov  1 21:30 requirements.txt
drwxrwxr-x  4 ubuntu ubuntu  4096 Nov  1 21:30 courses
-rw-rw-r--  1 ubuntu ubuntu  1173 Nov  1 21:30 .gitignore
-rw-rw-r--  1 ubuntu ubuntu 35141 Nov  1 21:30 LICENSE
-rw-rw-r--  1 ubuntu ubuntu   280 Nov  1 21:30 README.md
(fastai) ubuntu@ip-172-31-10-243:~/fastai$ 
```
```git pull```
```bash
(fastai) ubuntu@ip-172-31-10-243:~/fastai$ git pull
remote: Counting objects: 21, done.
remote: Total 21 (delta 12), reused 12 (delta 12), pack-reused 9
Unpacking objects: 100% (21/21), done.
From https://github.com/fastai/fastai
   9ae40be..d64a103  master     -> origin/master
Updating 9ae40be..d64a103
Fast-forward
 courses/dl1/excel/collab_filter.xlsx   | Bin 0 -> 90259 bytes
 courses/dl1/excel/conv-example.xlsx    | Bin 0 -> 101835 bytes
 courses/dl1/excel/entropy_example.xlsx | Bin 0 -> 10228 bytes
 courses/dl1/excel/graddesc.xlsm        | Bin 0 -> 124265 bytes
 courses/dl1/excel/layers_example.xlsx  | Bin 0 -> 17931 bytes
 courses/dl1/lesson1-rxt50.ipynb        |   4 +++-
 fastai/conv_learner.py                 |   5 +++--
 fastai/dataset.py                      |   2 ++
 fastai/imports.py                      |   1 +
 fastai/model.py                        |   4 +++-
 fastai/plots.py                        |  26 ++++++++++++++++++++++++++
 fastai/structured.py                   |   4 ++--
 fastai/torch_imports.py                |   4 ++++
 13 files changed, 44 insertions(+), 6 deletions(-)
 create mode 100755 courses/dl1/excel/collab_filter.xlsx
 create mode 100644 courses/dl1/excel/conv-example.xlsx
 create mode 100644 courses/dl1/excel/entropy_example.xlsx
 create mode 100644 courses/dl1/excel/graddesc.xlsm
 create mode 100644 courses/dl1/excel/layers_example.xlsx
(fastai) ubuntu@ip-172-31-10-243:~/fastai$ 
```


>my example
```
~/.ssh
▶ ssh -i aws_fastai_gpu.pem ubuntu@54.175.101.64 -L8888:localhost:8888       
Welcome to Ubuntu 16.04.3 LTS (GNU/Linux 4.4.0-1039-aws x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

  Get cloud support with Ubuntu Advantage Cloud Guest:
    http://www.ubuntu.com/business/services/cloud

2 packages can be updated.
0 updates are security updates.

*** System restart required ***
(fastai) ubuntu@ip-172-31-10-243:~$ 
```




And, now my [Lesson 1 notebook](https://s.users.crestle.com/u-fqnc8t2x12/notebooks/courses/fastai/courses/dl1/lesson1.ipynb) works!  :boom:



---

## Create a keypair  

#### Step 1:  go to appropriate directory in termainal
* In your Terminal, go to `.ssh` folder under your home directory  
(Note:  Windows users should have Ubuntu installed.)  
>my example
`/Users/reshamashaikh/.ssh`

#### Step 2:  create `id_rsa` files if needed
If you do not have these two files (`id_rsa` and `id_rsa.pub`), create them by typing:  
(Note:  this will create a special password for your computer to be able to log onto AWS.)  
`ssh-keygen`

Hit `<enter>` 3 times

#### Step 3:  import key files to AWS
(Note:  Extra step for Windows users:  you will need to copy these files to your hardrive from Ubuntu.)  
In AWS, go to **Key Pairs** in left menu and import `id_rsa`.  This step connects your local computer to AWS.  

 


