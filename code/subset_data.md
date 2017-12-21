# Subset data using `shuf`

From the directory of your notebook (from where you have the data folder available) run the following:
```bash
mkdir -p data/dogscats_sample/{valid,train}/{cats,dogs}
shuf -n 200 -e data/dogscats/train/cats | xargs -i cp {} data/dogscats_sample/train/cats
shuf -n 200 -e data/dogscats/train/cats/* | xargs -i cp {} data/dogscats_sample/train/cats
shuf -n 200 -e data/dogscats/train/dogs/* | xargs -i cp {} data/dogscats_sample/train/dogs
shuf -n 100 -e data/dogscats/valid/cats/* | xargs -i cp {} data/dogscats_sample/valid/cats
shuf -n 100 -e data/dogscats/valid/dogs/* | xargs -i cp {} data/dogscats_sample/valid/dogs
```

```
ls camels | wc -l 
```

```bash
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ cp horses/*.jpeg train/horses/
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ cp horses/*.jpeg valid/horses/
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ cp camels/*.jpeg train/camels/
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ cp camels/*.jpeg valid/camels/
```
```bash
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls horses | wc -l 
101
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls train/horses | wc -l
101
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls valid/horses | wc -l
101
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls camels | wc -l 
101
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls train/camels | wc -l
101
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls valid/camels | wc -l
101
```

```bash
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls ~/data/camelshorses/camels | wc -l
101
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls ~/data/camelshorses/horses | wc -l
101
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls ~/data/camelshorses/train | wc -l
2
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls ~/data/camelshorses/train/camels | wc -l
0
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls ~/data/camelshorses/train/horses | wc -l
0
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls ~/data/camelshorses/valid/camels | wc -l
0
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ ls ~/data/camelshorses/valid/horses | wc -l
0
(fastai) ubuntu@ip-172-31-2-59:~/data/camelshorses$ 
```

In your notebook, change the PATH to PATH = "data/dogscats_sample/"
The awesome command @jeremy shared on Twitter was this (please note the mv that you want to normally 
use when creating the train / valid / test splits):

shuf -n 5000 -e all/*.* | xargs -i mv {} all_val/
