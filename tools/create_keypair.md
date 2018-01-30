# Create a keypair  

### Step 1:  go to appropriate directory in termainal
* In your Terminal, go to `.ssh` folder under your home directory  
(Note:  Windows users should have Ubuntu installed.)  
>my example
`/Users/reshamashaikh/.ssh`

**Note:**  If you do not have the `.ssh` directory, you can create it (make sure you are in your home directory):  
`mkdir .ssh` 

### Step 2:  create `id_rsa` files if needed
**Note:**  these `id_rsa` files contain a special password for your computer to be able to log onto AWS.

If you do not have these two files (`id_rsa` and `id_rsa.pub`), create them by typing:  
- `ssh-keygen`
- Hit `<enter>` 3 times

>my example
```bash
% pwd 
/Users/reshamashaikh/.ssh
% ls
% ssh-keygen
Generating public/private rsa key pair.
Enter file in which to save the key (/Users/reshamashaikh/.ssh/id_rsa): 
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in /Users/reshamashaikh/.ssh/id_rsa.
Your public key has been saved in /Users/reshamashaikh/.ssh/id_rsa.pub.
The key fingerprint is:
SHA256:jmDJes1qOzDi8KynXLGQ098JMSRnbIyt0w7vSgEsr2E reshamashaikh@RESHAMAs-MacBook-Pro.local
The key's randomart image is:
+---[RSA 2048]----+
|   .=+           |
|.  .==           |
|.o  +o           |
|..+= oo          |
|.E.+X.  S        |
|+o=o=*oo.        |
|++.*o.+o.        |
|..*.oo           |
|o= o+o           |
+----[SHA256]-----+
% ls
total 16
-rw-------  1   1675 Dec 17 12:20 id_rsa
-rw-r--r--  1    422 Dec 17 12:20 id_rsa.pub
% 
```

### Step 3:  import key files to AWS
(Note:  Extra step for Windows users:  you will need to copy these files to your hardrive from Ubuntu.)  
In AWS, go to **Key Pairs** in left menu and import `id_rsa.pub`.  This step connects your local computer to AWS.  
Note for Mac Users:  can also `cat id_rsa.pub` in terminal, copy and paste it into AWS for "key contents".


 
