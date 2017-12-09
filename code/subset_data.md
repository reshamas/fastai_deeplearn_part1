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

In your notebook, change the PATH to PATH = "data/dogscats_sample/"
The awesome command @jeremy shared on Twitter was this (please note the mv that you want to normally 
use when creating the train / valid / test splits):

shuf -n 5000 -e all/*.* | xargs -i mv {} all_val/
