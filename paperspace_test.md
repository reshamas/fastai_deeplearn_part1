

## Simplifying Logging in to Paperspace
### (skip password and updating IP address in Jupyter Notebook link)
Note:  the below commands are run on **your local computer**

### Step 1:  Install `ssh-copy-id`
If you don't have it already, here's how to install it (Mac):  
```bash
brew install ssh-copy-id
```

### Step 2:  Ensure public keys are available
- `cd` into `~/.ssh` directory
- if you don't have an `.ssh` directory in your home folder, create it (`mkdir ~/.ssh`)
- if you don't have an `id_rsa.pub` file in your `~/.ssh` folder, create it (`ssh-keygen` and hit <kbd> Enter </kbd> 3 times)

### Step 3:  Copy public key to Paperspace
- replace IP address in syntax below with your own, and run command
```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub paperspace@184.105.6.151
```
### Step 4:  Add Paperspace info to `config` file
- make sure you are in the right directory
```bash
cd ~/.ssh
```

- if you don't have a `config` file, create one
```bash
nano config
```

- add these contents to you config file (replace IP address here with your Paperspace IP address)
```text
Host paperspace
     HostName 184.105.2.222
     IdentityFile ~/.ssh/id_rsa
     # StrictHostKeyChecking no  
     User paperspace
```
- here's the nano command for saving file  
<kbd> ctrl o </kbd>  
<kbd> <enter> </kbd>  

- here's the nano command for exiting a file  
<kbd> ctrl x </kbd>
>my example of config file
```bash
% pwd
/Users/reshamashaikh/.ssh
% cat config
Host paperspace
     HostName 184.105.2.222
     IdentityFile ~/.ssh/id_rsa
     # StrictHostKeyChecking no  
     User paperspace
```

## Back to Paperspace
- go to `fastai` directory and launch Jupyter Notebook
```bash
cd fastai
```
```bash
jupyter notebook
```

>my example
```bash
Last login: Sun Jan  7 12:57:35 2018 from 77.777.777.777
(fastai) paperspace@psgyqmt1m:~$ ls
anaconda3  data  downloads  fastai
(fastai) paperspace@psgyqmt1m:~$ cd fastai
(fastai) paperspace@psgyqmt1m:~/fastai$ ls
CODE-OF-CONDUCT.md  environment.yml  LICENSE   MANIFEST.in  README.rst        setup.cfg  tutorials
courses             fastai           MANIFEST  README.md    requirements.txt  setup.py

(fastai) paperspace@psgyqmt1m:~/fastai$ jupyter notebook
[I 12:58:13.608 NotebookApp] Writing notebook server cookie secret to /run/user/1000/jupyter/notebook_cookie_secret
[W 12:58:14.363 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
[I 12:58:14.376 NotebookApp] Serving notebooks from local directory: /home/paperspace/fastai
[I 12:58:14.376 NotebookApp] 0 active kernels
[I 12:58:14.376 NotebookApp] The Jupyter Notebook is running at:
[I 12:58:14.376 NotebookApp] http://[all ip addresses on your system]:8888/?token=594036202395d8ea6324d33ecee448cd87e99a50b64918cb
[I 12:58:14.376 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 12:58:14.377 NotebookApp] 
    
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=77594036202395d8ea6324d33ecee448cd87e99a50b64918cb        
```

## Cool!  no password required, and no need to update notebook url with IP address
- this is my url link, and it works! :boom:
http://localhost:8888/?token=77594036202395d8ea6324d33ecee448cd87e99a50b64918cb
