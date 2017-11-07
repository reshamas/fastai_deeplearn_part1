# Create a keypair  

### Step 1:  go to appropriate directory in termainal
* In your Terminal, go to `.ssh` folder under your home directory  
(Note:  Windows users should have Ubuntu installed.)  
>my example
`/Users/reshamashaikh/.ssh`

### Step 2:  create `id_rsa` files if needed
If you do not have these two files (`id_rsa` and `id_rsa.pub`), create them by typing:  
(Note:  this will create a special password for your computer to be able to log onto AWS.)  
`ssh-keygen`

Hit `<enter>` 3 times

### Step 3:  import key files to AWS
(Note:  Extra step for Windows users:  you will need to copy these files to your hardrive from Ubuntu.)  
In AWS, go to **Key Pairs** in left menu and import `id_rsa`.  This step connects your local computer to AWS.  

 
