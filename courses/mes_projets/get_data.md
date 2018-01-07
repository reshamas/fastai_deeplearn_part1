# Iceberg

```bash
(fastai) ubuntu@ip-172-31-2-59:~/data$ pwd
/home/ubuntu/data
(fastai) ubuntu@ip-172-31-2-59:~/data$ 
````

https://www.kaggle.com/c/statoil-iceberg-classifier-challenge


```bash
rm ~/.kaggle-cli/browser.pickle
pip install kaggle-cli --upgrade
```

```bash
kg download -u "reshamashaikh" -p "xxx" -c statoil-iceberg-classifier-challenge
```

```bash
sudo apt-get install p7zip-full
7z e test.json.7z
7z e train.json.7z 
7z e sample_submission.csv.7z 
```

```
(fastai) ubuntu@ip-172-31-2-59:~/data/iceberg$ ls -alt
total 1972980
drwxrwxr-x 2 ubuntu ubuntu       4096 Jan  7 20:44 .
drwxrwxr-x 5 ubuntu ubuntu       4096 Jan  7 20:38 ..
-rw-rw-r-- 1 ubuntu ubuntu  257127394 Jan  7 20:36 test.json.7z
-rw-rw-r-- 1 ubuntu ubuntu   44932785 Jan  7 20:36 train.json.7z
-rw-rw-r-- 1 ubuntu ubuntu      38566 Jan  7 20:36 sample_submission.csv.7z
-rw-rw-r-- 1 ubuntu ubuntu     117951 Oct 23 17:27 sample_submission.csv
-rw-rw-r-- 1 ubuntu ubuntu 1521771850 Oct 23 17:27 test.json
-rw-rw-r-- 1 ubuntu ubuntu  196313674 Oct 23 17:23 train.json
(fastai) ubuntu@ip-172-31-2-59:~/data/iceberg$ 
```
```bash
(fastai) ubuntu@ip-172-31-2-59:~/data/iceberg$ wc -l *
      8425 sample_submission.csv
       151 sample_submission.csv.7z
         0 test.json
   1004794 test.json.7z
         0 train.json
    175531 train.json.7z
   1188901 total
(fastai) ubuntu@ip-172-31-2-59:~/data/iceberg$
```



