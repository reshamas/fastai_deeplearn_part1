# Setting Up a Personal Deep Learning Computer
These are steps to replicate the AWS AMI setup on your own computer (assuming you have NVIDIA CUDA GPUs)  
Recommended hardware:  NVIDIA GTX-1080 Ti  


## Step 1: Install Anaconda Python 3.6
Python version should be 3.6+  
https://conda.io/docs/user-guide/tasks/manage-python.html


## Step 2: Clone the fastai library
```git
git clone https://github.com/fastai/fastai.git
```

## Step 3: Go to directory where `environment.yml` file
- The environment.yml file is under this directory.  
- The environment file contains all the dependencies needed:  https://github.com/fastai/fastai/blob/master/environment.yml

```bash
cd fastai/ 
```

## Step 4: Install header files to build python extensions
This step installs python3-dev package.
```bash
sudo apt-get install python3-dev
```

## Step 5:  Create the virtual environment
This step installs all of the dependencies.  
```bash
conda env create -f environment.yml
```

## Step 6:  Activate virtual environment 
Do this step every time you login. Or else put it in your `.bashrc` file.  
```bash
source activate fastai
```
