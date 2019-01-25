# Camels vs Horses

## Important Links
- [Google Cloud Platform](http://course-v3.fast.ai/start_gcp.html)
  - [GCP: update fastai, conda & packages](http://course-v3.fast.ai/start_gcp.html#step-4-access-fastai-materials-and-update-packages)

---

# Downloading Images
[Fastai tutorial: downloading images](https://github.com/fastai/course-v3/blob/master/nbs/dl1/download_images.ipynb)
- After this step in the Chrome Javascript console:  

```java
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
```
in Mac, it will download a file called `download.csv` to my `~/Downloads` folder
- rename the folder to your image.  For me:  
1.  camels.csv
2.  horses.csv

#### Go to my `Downloads` directory
```bash
pwd
```
```
/Users/reshamashaikh/Downloads
```

#### List items in directory in reverse order
```bash
ls -lrt
```
```
-rw-r--r--@   1        68354 Oct 26 17:01 camels.csv
-rw-r--r--@   1        85497 Oct 26 17:03 horses.csv
```

## `scp` to GCP
```bash
gcloud compute scp camels.csv jupyter@my-fastai-instance:~
gcloud compute scp horses.csv jupyter@my-fastai-instance:~
```

## on GCP: move data to `data` directory
```bash
jupyter@my-fastai-instance:~$ ls
camels.csv  course-v3  horses.csv  tutorials
jupyter@my-fastai-instance:~$ mv *.csv /home/jupyter/tutorials/data
jupyter@my-fastai-instance:~$ ls
course-v3  tutorials
jupyter@my-fastai-instance:~$ ls tutorials/data
camels.csv  horses.csv
jupyter@my-fastai-instance:~$ 
```

## convert `.csv` files to `.txt` files
```bash
cat camels.csv | tr  ',' '\n' > camels.txt
cat horses.csv | tr  ',' '\n' > horses.txt
```

## rename files to match notebook
```bash
mv camels.txt urls_camels.txt
mv horses.txt urls_horses.txt
```

## Create directory and upload urls file into your server
- Original [notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/download_images.ipynb
```bash
my_path = "/home/jupyter/tutorials/"
```
```bash
folder = 'camels'
file = 'urls_camels.txt'
```
```bash
path = Path(my_path+'data/mammals')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
```
do same for "horses"

### Move url_name.txt file to appropriate folder
```bash
mv urls_camels.txt /home/jupyter/tutorials/data/mammals/camels
mv urls_horses.txt /home/jupyter/tutorials/data/mammals/horses
```

## Directory Tree
```bash
jupyter@my-fastai-instance:~/tutorials/data$ pwd
/home/jupyter/tutorials/data
jupyter@my-fastai-instance:~/tutorials/data$ tree -d
.
└── mammals
    ├── camels
    └── horses

3 directories
jupyter@my-fastai-instance:~/tutorials/data$
```

## let's look at file
```bash
head urls_camels.txt 
```

```bash
jupyter@my-fastai-instance:~/tutorials/data/mammals$ head urls_camels.txt 
https://media.buzzle.com/media/images-en/gallery/mammals/camels/1200-close-up-of-camel-nostrils.jpg
http://www.cidrap.umn.edu/sites/default/files/public/styles/ss_media_popup/public/media/article/baby_camel_nursing.jpg?itok=0vwqXyoW
https://www.thenational.ae/image/policy:1.632918:1506081168/image/jpeg.jpg?f=16x9&w=1200&$p$f$w=dfa40e8
https://i.dailymail.co.uk/i/pix/2012/11/24/article-2237967-162CA49A000005DC-153_634x409.jpg
https://samslifeinjeddah.files.wordpress.com/2014/08/jed-camel-2_edited.jpg
https://i.pinimg.com/236x/29/94/04/299404d417dd8b836b4a5c396cb597a6--camel-animal-baby-camel.jpg
https://i.chzbgr.com/full/9056188416/h8763E301/
https://i.dailymail.co.uk/i/pix/2012/11/24/article-2237967-162CA5A0000005DC-2_634x372.jpg
https://secure.i.telegraph.co.uk/multimedia/archive/01676/Camel_Milk_1676595c.jpg
https://upload.wikimedia.org/wikipedia/commons/4/43/07._Camel_Profile%2C_near_Silverton%2C_NSW%2C_07.07.2007.jpg
jupyter@my-fastai-instance:~/tutorials/data/mammals$ 
```

