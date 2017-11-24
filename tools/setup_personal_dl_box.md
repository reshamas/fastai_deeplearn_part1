# Setting Up Personal DL Computer
These are steps to replicate the AWS AMI setup on your own computer (assuming you have CUDA GPUs, etc)  

### Step 1: Install Anaconda 3
Anaconda version should be 3.6+

### Step 2: Clone the fastai library
```python
git clone https://github.com/fastai/fastai.git
```

### Step 3: Go to directory where `environment.yml` file
```bash
cd fastai/ 
```
the environment.yml file is under this directory

### Step 4:  Create the environment
```bash
conda env create -f environment.yml
```

### Step 5:  Activate virtual environment 
```
source activate fastai
```
You need to do that step every time you login. Or else put it in your `.bashrc` file
