# Fastai FAQs for Beginners

## Q1:  Where can I put _my_ Jupter Notebook?

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

file:  `.bashrc`  
```
export PYTHONPATH=$PYTHONPATH:~/fastai
```  
**Reminder:*** don't forget to run (or `source`) your `.bashrc` file:  
1.  add path to `.bashrc`
2.  save and exit
3.  source it:  `source ~/.bashrc`

