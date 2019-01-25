# Logging in to GCP

## Step 1:  GCP Console
1.  Go to my [GCP console](https://console.cloud.google.com/compute/instances?project=near-earth-comets-f8c3f&folder&organizationId&duration=PT1H)
2.  `Start` my instance, which is called `my-fastai-instance`

## Step 2:  My Mac Terminal
0.  `gcloud auth login`  May need to login in via Google chrome

1.  Go to my terminal on the Mac, type this:  
```bash
gcloud compute ssh --zone=$ZONE jupyter@my-fastai-instance -- -L 8080:localhost:8080
```
```bash
gcloud compute ssh --zone=us-west2-b jupyter@my-fastai-instance -- -L 8080:localhost:8080
```

>Enter passphrase for key '/Users/reshamashaikh/.ssh/google_compute_engine': 
```
xxxxx
```
I will see this:  
```bash
jupyter@my-fastai-instance:~$ 
```

## Updating
### Important Links
- [Google Cloud Platform](http://course-v3.fast.ai/start_gcp.html)
  - [GCP: update fastai, conda & packages](http://course-v3.fast.ai/start_gcp.html#step-4-access-fastai-materials-and-update-packages)

### Updating packages
```bash
cd course-v3/
git pull
```
```bash
jupyter@my-fastai-instance:~/course-v3$ cd ..
jupyter@my-fastai-instance:~$ pwd
/home/jupyter
```
```bash
cd tutorials/fastai
git checkout .
git pull
```

## Update fastai library
```bash
sudo /opt/anaconda3/bin/conda install -c fastai fastai
```
```bash
conda install -c fastai fastai
```

### get fastai version ---> in terminal
```bash
pip list | grep fastai
```



---
```bash
jupyter@my-fastai-instance:~/tutorials/fastai$ pip list | grep fastai
fastai                             1.0.12
```
Fri, 11/12/18
```bash
fastai                             1.0.18 
```
Sat, 12/8/18
```bash
fastai                             1.0.35
```
Sat, 12/15/18
```bash
From https://github.com/fastai/fastai
   7d617eda..af59fa03  master         -> origin/master
 * [new branch]        release-1.0.36 -> origin/release-1.0.36
 * [new branch]        release-1.0.37 -> origin/release-1.0.37
 * [new tag]           1.0.37         -> 1.0.37
 * [new tag]           1.0.36         -> 1.0.36
 * [new tag]           1.0.36.post1   -> 1.0.36.post1
```

### get fastai version ---> in Jupyter notebook
```python
import torch
print(torch.__version__)
import fastai
print(fastai.__version__)
```

## Step 3:  Get to Jupyter Notebook
- Go to localhost (run Jupyter Notebook):  
http://localhost:8080/tree

## Where am I working?
```bash
jupyter@my-fastai-instance:~/projects$ pwd
/home/jupyter/projects
```
http://localhost:8080/tree/projects


## Step 4:  Shut down GCP instance in the console
- Go to GCP console

---

- `ImageBunch`
- `TextDataBunch`
