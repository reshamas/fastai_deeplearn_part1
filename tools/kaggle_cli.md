# Kaggle CLI
(**CLI** = **C**ommand **L**ine **I**nterface)  

## Resource
[Kaggle CLI Wiki](http://wiki.fast.ai/index.php/Kaggle_CLI)

## Installation
Check to see if `kaggle-cli` is installed:  
<kbd> kaggle-cli --version </kbd>  

Install `kaggle-cli`:  
<kbd> pip install kaggle-cli </kbd>   

May need to **update package** if you run into errors:  
<kbd> pip install kaggle-cli --upgrade </kbd>


---

## [Kaggle Competition Datasets](https://www.kaggle.com/datasets)
Note 1:  You must have a Kaggle user ID and password.  If you logged in to Kaggle using FB or LI, you'll have to reset your password, as that is needed for command line access to the data.  

Note 2:  Pick a competition, and ensure you have **accepted the rules** of that competition.  Otherwise, you will not be able to download the data using the CLI.



### Step 1:  Identify the competition I will use
https://www.kaggle.com/c/dogs-vs-cats   

**Note:**  the competition name can be found in the url; here it is **dogs-vs-cats**

### Step 2:  Accept competition rules  
https://www.kaggle.com/c/dogs-vs-cats/rules

### Step 3:  Set up data directory 
<kbd> ls </kbd>  
<kbd> mkdir data </kbd>  
<kbd> cd data </kbd>  
>my example
```bash
ubuntu@ip-10-0-0-13:~$ ls
anaconda2  anaconda3  downloads  git  nbs  temp
ubuntu@ip-10-0-0-13:~$ mkdir data
ubuntu@ip-10-0-0-13:~$ cd data
```

### Step 4a:  Download data (try 1)
Syntax:  
<kbd> config -g -u 'username' -p 'password' -c 'competition' </kbd>  
<kbd> kg download </kbd>  

Note:  Here's an example of warning message I receive when I tried to download data before accepting the rules of the competition:  
>my example
```bash
ubuntu@ip-10-0-0-13:~/data$ kg config -g -u 'reshamashaikh' -p 'xxx' -c dogs-vs-cats
ubuntu@ip-10-0-0-13:~/data$ kg download
Starting new HTTPS connection (1): www.kaggle.com
downloading https://www.kaggle.com/c/dogs-vs-cats/download/sampleSubmission.csv

sampleSubmission.csv N/A% |                                                                                                                   | ETA:  --:--:--   0.0 s/B

Warning: download url for file sampleSubmission.csv resolves to an html document rather than a downloadable file. 
Is it possible you have not accepted the competition's rules on the kaggle website?
```

### Step 4b:  Dowload data (try 2)
Note 1:  I have accepted the competition rules; will try downloading again   
<kbd> config -g -u 'username' -p 'password' -c 'competition' </kbd>  
<kbd> kg download </kbd>  
>my example
```bash
ubuntu@ip-10-0-0-13:~/data$ kg config -g -u 'reshamashaikh' -p 'xxx' -c dogs-vs-cats
ubuntu@ip-10-0-0-13:~/data$ kg download
Starting new HTTPS connection (1): www.kaggle.com
downloading https://www.kaggle.com/c/dogs-vs-cats/download/sampleSubmission.csv

Starting new HTTPS connection (1): storage.googleapis.com
sampleSubmission.csv 100% |##################################################################################################################| Time: 0:00:00 320.2 KiB/s

downloading https://www.kaggle.com/c/dogs-vs-cats/download/test1.zip

test1.zip 100% |#############################################################################################################################| Time: 0:00:08  32.5 MiB/s

downloading https://www.kaggle.com/c/dogs-vs-cats/download/train.zip

train.zip 100% |#############################################################################################################################| Time: 0:00:17  31.4 MiB/s
```
### Step 5:  Look at data that was downloaded
<kbd> ls -alt </kdb>  
```bash
ubuntu@ip-10-0-0-13:~/data$ ls -alt
total 833964
-rw-rw-r--  1 ubuntu ubuntu 569546721 Nov  4 18:24 train.zip
drwxrwxr-x  2 ubuntu ubuntu      4096 Nov  4 18:24 .
-rw-rw-r--  1 ubuntu ubuntu 284321224 Nov  4 18:24 test1.zip
-rw-rw-r--  1 ubuntu ubuntu     88903 Nov  4 18:23 sampleSubmission.csv
drwxr-xr-x 22 ubuntu ubuntu      4096 Nov  4 18:23 ..
ubuntu@ip-10-0-0-13:~/data$ 
```

### Step 6:  Unzip Files
Note 1:  You will need to install and use `unzip` to unzip files.

<kbd> sudo apt install unzip </kbd>  
<kbd> unzip train.zip </kbd>  
<kbd> unzip -q test.zip </kbd>  (Note:  `-q` means to unzip quietly, suppressing the printing)  

```bash
ubuntu@ip-10-0-0-13:~/nbs/data$ ls train/dogs/dog.1.jpg
train/dogs/dog.1.jpg
ubuntu@ip-10-0-0-13:~/nbs/data$ ls -l train/dogs/ | wc -l
12501
ubuntu@ip-10-0-0-13:~/nbs/data$ 


ubuntu@ip-10-0-0-13:~/nbs/data$ ls -l train/cats/ | wc -l
12501
ubuntu@ip-10-0-0-13:~/nbs/data$
ubuntu@ip-10-0-0-13:~/nbs/data$ ls test1 | wc -l
12500
ubuntu@ip-10-0-0-13:~/nbs/data$ 
```

### Jeremy’s Setup
Good to copy 100 or so the sample directory; enough to check that the scripts are working

Advice 1:  Separate TEST data into VALIDATION
TASK:  move 1000 each dogs / cats into valid 
```bash
> ls valid/cats/ | wc -l
1000
> ls valid/dogs/ | wc -l
1000

Advice 2:  Do all of your work on sample data
> ls sample/train

> ls sample/valid

> ls sample/train/cats | wc -l
8
> ls sample/valid/cats | wc -l
4
```
