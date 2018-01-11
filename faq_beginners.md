# Fastai FAQs for Beginners

### Q:  Where can I put _my_ Jupter Notebook?

#### Option 1 (default):  under /courses
The default location is under the `dl1` folder, wherever you've cloned the repo on your GPU machine.
>my example
```bash
(fastai) paperspace@psgyqmt1m:~$ ls
anaconda3  data  downloads  fastai
```

#### Option 2:  where you want
If you change the default location of your notebook, you'll need to update your `.bashrc` file.  Add the path to where you've cloned the GitHub repo:  
- for me, it is at the root level (as in Paperspace or AWS)
- but, it could be, for you, `~/User/ubuntu/git_repos/fastai`

file:  `.bashrc`  
```
export PYTHONPATH=$PYTHONPATH:~/fastai
```  
**Reminder:*** don't forget to run (or `source`) your `.bashrc` file:  
1.  add path to `.bashrc`
2.  save and exit
3.  source it:  `source ~/.bashrc`

