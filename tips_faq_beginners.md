# Fastai FAQs for Beginners

## Q1:  How to ask for help for fastai
http://wiki.fast.ai/index.php/How_to_ask_for_Help

---
## Q2:  Where can I put _my_ Jupter Notebook?

:red_circle: **NOTE:** Do NOT put your Jupyter Notebook under the `/data/` directory!  Here's [the link](http://forums.fast.ai/t/how-to-remove-ipynb-checkpoint/8532/2) for why.

### Option 1 (default):  under /courses
The default location is under the `dl1` folder, wherever you've cloned the repo on your GPU machine.
>my example
```bash
(fastai) paperspace@psgyqmt1m:~$ ls
anaconda3  data  downloads  fastai
```
- Paperspace:  `/home/paperspace/fastai/courses/dl1`
- AWS:         `/home/ubuntu/fastai/courses/dl1`

### Option 2:  where you want
If you change the default **location of your notebook**, you'll need to update your `.bashrc` file.  Add in the path to where you've cloned the fastai GitHub repo:  
- for me, my notebooks are in a "projects" directory:  `~/projects`
- my `fastai` repo is cloned at the root level, so it is here:  `~/fastai`

in the file `.bashrc`  add this path:
```
export PYTHONPATH=$PYTHONPATH:~/fastai
```  
**Reminder:** don't forget to run (or `source`) your `.bashrc` file:  
1.  add path where fastai repo is to `.bashrc`
2.  save and exit
3.  source it:  `source ~/.bashrc`

### Option 3:  used `pip install`
Note that if you did `pip install`, you don't need to specify the path (as in option 2, or you don't need to put in the courses folder, as in option 1).  
However, fastai is still being updated so there is a delay in library available in pip

---
## Q3:  What does my directory structure look like?
>my path
```bash
PATH = "/home/ubuntu/data/dogscats/"
```

>looking at my directory structure
```bash
!tree {PATH} -d
```
```bash
/home/ubuntu/data/dogscats/
├── models
├── sample
│   ├── models
│   ├── tmp
│   ├── train
│   │   ├── cats
│   │   └── dogs
│   └── valid
│       ├── cats
│       └── dogs
├── test1
├── train
│   ├── cats
│   └── dogs
└── valid
    ├── cats
    └── dogs
```
### Notes on directories
* `models` directory:  created automatically
* `sample` directory:  you create this with a small sub-sample, for testing code
* `test` directory:  put any test data there if you have it
* `train`/`test` directory:  you create these and separate the data using your own data sample
* fastai / keras code automatically picks up the **label** of your categories based on your folders.  Hence, in this example, the two labels are:  dogs, cats

### Notes on image file names
* not important, you can name them whatever you want


>looking at file counts
```bash
# print number of files in each folder

print("training data: cats")
!ls -l {PATH}train/cats | grep ^[^d] | wc -l

print("training data: dogs")
!ls -l {PATH}train/dogs | grep ^[^d] | wc -l

print("validation data: cats")
!ls -l {PATH}valid/cats | grep ^[^d] | wc -l

print("validation data: dogs")
!ls -l {PATH}valid/dogs | grep ^[^d] | wc -l

print("test data")
!ls -l {PATH}test1 | grep ^[^d] | wc -l
```
>my output
```bash
training data: cats
11501
training data: dogs
11501
validation data: cats
1001
validation data: dogs
1001
test data
12501
```
