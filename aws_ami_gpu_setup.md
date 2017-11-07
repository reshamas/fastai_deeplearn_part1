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

#### Finish:  log into our AWS instance
* Get the PublicIP address of your AWS computer
* `ssh` in to instance from your local computer
>my example
```bash




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

 


