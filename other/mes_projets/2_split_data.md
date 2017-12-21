
```bash
# make the sub directories
mkdir -p data/camelhorse/{train,valid}/{camel,horse}

# split original data into train/test
shuf -n 68 -e data/camelhorse/camels/* | xargs -i cp {} data/camelhorse/train/camel
shuf -n 68 -e data/camelhorse/horses/* | xargs -i cp {} data/camelhorse/train/horse
shuf -n 33 -e data/camelhorse/camels/* | xargs -i cp {} data/camelhorse/valid/camel
shuf -n 33 -e data/camelhorse/horses/* | xargs -i cp {} data/camelhorse/valid/horse


ls ~/data/camelhorse/camels | wc -l
ls ~/data/camelhorse/horses | wc -l
ls ~/data/camelhorse/train/camel | wc -l
ls ~/data/camelhorse/train/horse | wc -l
ls ~/data/camelhorse/valid/camel | wc -l
ls ~/data/camelhorse/valid/horse | wc -l
```
