# splitting data
# this is an example of camels/horses dataset
# ls camels | wc -l 
# 0.  shuffle the data
#     split the data into train/valid
# 1.  split the data for us into train/valid
# 2.  take a subset and split that into train/valid

sample_t=${1:-50}
sample_v=${2:-20}
data_f="camelshorses"
sample_f="camelshorses_sample"
category_1="camels"
category_2="horses"

n_t=65
n_v=35

ns_t=40
ns_v=20
outfile="shuffled.txt"

# create the folders
mkdir -p data/$data_f/{train,valid}/{$category_1,$category_2}
mkdir -p data/$sample_f/{train,valid}/{$category_1,$category_2}

# shuffle the data for LABEL #1
# copy into train and valid folders

echo "print contents of shuffled.txt"
echo "  "
cat /tmp/shuffled.txt

start_dir="data/$data_f/train/$category_1/*"
#shuf -e data/$data_f/train/$category_1/ >/tmp/shuffled.txt
shuf -e $start_dir > /tmp/shuffled.txt
	
echo "print contents of shuffled.txt"
echo "  "


cat /tmp/shuffled.txt

echo "copying images to train/valid for full dataset"

head /tmp/shuffled.txt -n $n_t | xargs -i cp {} data/$data_f/train/$category_1
tail /tmp/shuffled.txt -n $n_v | xargs -i cp {} data/$data_f/valid/$category_1

echo "copying images to train/valid for subset of data"
head /tmp/shuffled.txt -n $ns_t | xargs -i cp {} data/$sample_f/train/$category_1
tail /tmp/shuffled.txt -n $ns_v | xargs -i cp {} data/$sample_f/valid/$category_1


# shuffle the data for LABEL #1
# copy into train and valid folders
#shuf -e data/$data_f/train/$category_2/* > /tmp/shuffled2.txt

#head /tmp/shuffled2.txt -n $n_t | xargs -i cp {} data/$data_f/train/$category_2
#tail /tmp/shuffled2.txt -n $n_v | xargs -i cp {} data/$data_f/valid/$category_2

#head /tmp/shuffled2.txt -n $ns_t | xargs -i cp {} data/$sample_f/train/$category_2
#tail /tmp/shuffled2.txt -n $ns_v | xargs -i cp {} data/$sample_f/valid/$category_2




#shuf -n $n_t -e data/$data_f/$category_1/* | xargs -i cp {} data/$sample_f/train/$category_1
#shuf -n $n_t -e data/$data_f/$category_2/* | xargs -i cp {} data/$sample_f/train/$category_2
#shuf -n $n_v -e data/$data_f/$category_1/* | xargs -i cp {} data/$sample_f/valid/$category_1
#shuf -n $n_v -e data/$data_f/$category_2/* | xargs -i cp {} data/$sample_f/valid/$category_2



#mkdir -p data/$sample_f/{train,valid}/{$category_1,$category_2}
#shuf -n 200 -e data/dogscats/train/cats   | xargs -i cp {} data/dogscats_sample/train/cats

#shuf -n $sample_t -e data/$data_f/train/$category_1/* | xargs -i cp {} data/$sample_f/train/$category_1
#shuf -n $sample_t -e data/$data_f/train/$category_2/* | xargs -i cp {} data/$sample_f/train/$category_2
#shuf -n $sample_v -e data/$data_f/valid/$category_1/* | xargs -i cp {} data/$sample_f/valid/$category_1
#shuf -n $sample_v -e data/$data_f/valid/$category_2/* | xargs -i cp {} data/$sample_f/valid/$category_2
