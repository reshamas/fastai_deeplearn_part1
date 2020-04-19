
## Paperspace
- fastai: [Getting Started with Gradient](https://course.fast.ai/start_gradient.html)
- fastai: v4 [Paperspace (free, paid options)](https://forums.fast.ai/t/platform-paperspace-free-paid-options/65515)

### My steps on Paperspace
1.  notebook:  https://www.paperspace.com/telmjtws3/notebook/prjrrhy56
2.  Open terminal, via Jupyter Notebook
- type `bash` to get a regular terminal (autocomplete, etc)
- `pip install fastai2 fastcore --upgrade`
- `cd course-v4`
- `git pull`

### Back to work
1.  Log in:  https://www.paperspace.com
2.  To "notebooks" or "workspace":  https://www.paperspace.com/console/notebooks
3.  Actions / Start
4.  Actions / Open
5.  New / terminal

## updating packages on Paperspace
```bash
apt-get update
```
```bash
apt-get install libsndfile1-dev
```

## unzip files
```
   10  cd storage
   11  ls
   12  cd fowl_data/
   13  ls
   14  unzip Test.zip
   15  pwd
   16  clear
   17  history
```
```bash
root@6c4a45f4bab8:/notebooks/storage/fowl_data# unzip -q Train.zip
```


## Adding a data folder and data

6.  use bash shell:  `# bash`
7.  going to `storage` folder
```bash
root@51ae9bcde285:/notebooks/storage# pwd
/notebooks/storage
```
8.  can `mkdir` here to add datasets
```bash
# bash
root@51ae9bcde285:/notebooks# ls
course-v4  datasets  storage
root@51ae9bcde285:/notebooks# cd storage
root@51ae9bcde285:/notebooks/storage# ls
archive  data  models
root@51ae9bcde285:/notebooks/storage# mkdir fowl
```
9.  go to that directory
```bash
root@51ae9bcde285:/notebooks/storage# cd fowl
root@51ae9bcde285:/notebooks/storage/fowl# ls
root@51ae9bcde285:/notebooks/storage/fowl# pwd
/notebooks/storage/fowl
```
Tried:  `wget` and `curl` but urls were not working  
Zindi Fowl competition: https://zindi.africa/competitions/fowl-escapades/data

10. Go to Jupyter notebook in Paperspace
- navigate to `storage` folder
- use **upload** to upload files

## Data
```bash
root@3b9d9da72ac6:/notebooks/storage/fowl_data# pwd
/notebooks/storage/fowl_data
root@3b9d9da72ac6:/notebooks/storage/fowl_data# ls -alt
total 2104240
drwxr-xr-x 6 root root       4096 Mar 31 19:48 ..
-rw-r--r-- 1 root root 1407124233 Mar 31 16:26 Train.zip
-rw-r--r-- 1 root root  743620991 Mar 31 16:12 Test.zip
drwxr-xr-x 3 root root       4096 Mar 31 15:12 .
drwxr-xr-x 2 root root       4096 Mar 31 15:12 .ipynb_checkpoints
-rw-r--r-- 1 root root    3815649 Mar 31 15:12 StarterNotebook.ipynb
-rw-r--r-- 1 root root       2391 Mar 31 15:11 authors.csv
-rw-r--r-- 1 root root      80027 Mar 31 15:11 SampleSubmission.csv
-rw-r--r-- 1 root root      48594 Mar 31 15:11 Train.csv
-rw-r--r-- 1 root root      13679 Mar 31 15:11 Test.csv
root@3b9d9da72ac6:/notebooks/storage/fowl_data#
```
### rename directories
```bash
root@6c4a45f4bab8:/notebooks/storage/fowl_data# mv Train/ train/
root@6c4a45f4bab8:/notebooks/storage/fowl_data# mv Test/ test/
root@6c4a45f4bab8:/notebooks/storage/fowl_data#
```
```bash
conda install -c conda-forge ffmpeg
```

