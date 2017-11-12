# Kaggle CLI

#### Kaggle CLI Wiki
http://wiki.fast.ai/index.php/Kaggle_CLI

#### Note:  need to accept competition rules  
https://www.kaggle.com/c/dogs-vs-cats/rules

<kbd> config -g -u 'username' -p 'password' -c 'competition' </kbd>
```bash
ubuntu@ip-10-0-0-13:~$ kg config -g -u 'reshamashaikh' -p 'xxx' -c dogs-vs-cats
ubuntu@ip-10-0-0-13:~$ kg download
Starting new HTTPS connection (1): www.kaggle.com
downloading https://www.kaggle.com/c/dogs-vs-cats/download/sampleSubmission.csv

sampleSubmission.csv N/A% |                                                                                                                   | ETA:  --:--:--   0.0 s/B

Warning: download url for file sampleSubmission.csv resolves to an html document rather than a downloadable file. 
Is it possible you have not accepted the competition's rules on the kaggle website?
```

SYNTAX:
<kbd> config -g -u 'username' -p 'password' -c 'competition' </kbd>

```bash
ubuntu@ip-10-0-0-13:~$ kg config -g -u 'reshamashaikh' -p 'xxx' -c dogs-vs-cats
ubuntu@ip-10-0-0-13:~$ ls
anaconda2  anaconda3  downloads  git  nbs  temp
ubuntu@ip-10-0-0-13:~$ mkdir data
ubuntu@ip-10-0-0-13:~$ cd data
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
#### look at data that was downloaded
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

### Unzip Files
You will need to install and use unzip to unzip.

<kbd> sudo apt install unzip </kbd>  
<kbd> unzip train.zip </kbd>  
<kdb> unzip -q test.zip </kbd>

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
