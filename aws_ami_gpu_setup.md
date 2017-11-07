# AWS fastami Setup
fastai.ai Part 1 v2  
Updated:  07-Nov-2017  

AMI (Amazon Machine Image):  a template for how your computer is created



Log into AWS Console:  http://console.aws.amazon.com/  
Select Service:  EC2  
Launch Instance

### Step 1:  Choose an Amazon Machine Image (AMI)
* Search Community AMIs [left menu]
* Search:  `fastai`
* Select this image (for region N. Virginia):  `fastai-part1v2-p2 - ami-c6ac1cbc`


### Step 2:  Choose an Instance Type
(Note:  it is the kind of computer we want to choose.)  
* Filter by:  `GPU Compute`
* Select:  `p2.xlarge`   (this is the cheapeast, reasonably effective for deep learning type of instance available)
* Select: `Review and Launch` at bottom

#### Step 2b:  Select keypair
Note:  you have already created a keypair in the past.  Use one of thoese.  For more specific instructions, see "How to Create a Keypair."

---
## Create a keypair  
* In your Terminal, go to `.ssh` folder under your home directory  
(Note:  Windows users should have Ubuntu installed.)  
>my example
`/Users/reshamashaikh/.ssh`

If you do not have these two files (`id_rsa` and `id_rsa.pub`), create them by typing:  
(Note:  this will create a special password so your computer can log onto AWS.)  
`ssh-keygen`



And, now my [Lesson 1 notebook](https://s.users.crestle.com/u-fqnc8t2x12/notebooks/courses/fastai/courses/dl1/lesson1.ipynb) works!  :boom:



