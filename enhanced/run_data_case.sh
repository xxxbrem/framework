#!/bin/bash

echo "Data Case, targetting data_batch_2:"
original_size=$(wc -c "./data/cifar-10-batches-py/data_batch_2" | awk '{print $1}')
echo "The Original Size of data_batch_2 is $original_size."
python main.py --file_type CIFAR-10 --file_path data/cifar-10-batches-py --output_path out
echo "The Size of the Compressed Clean data_batch_2 is $(wc -c "./out/data_batch_2" | awk '{print $1}')."
python main.py --file_type CIFAR-10 --file_path out1 --output_path out1
echo "The Size of the Compressed Poisoned data_batch_2 is $(wc -c "./out1/data_batch_2" | awk '{print $1}')."
echo "Compression Complete!"

mkdir ../hashclash/cpc_workdir1
cp out/data_batch_2 ../hashclash/cpc_workdir1
mv ../hashclash/cpc_workdir1/data_batch_2 ../hashclash/cpc_workdir1/data_batch_2_clean
cp out1/data_batch_2 ../hashclash/cpc_workdir1
mv ../hashclash/cpc_workdir1/data_batch_2 ../hashclash/cpc_workdir1/data_batch_2_poisoned
cd ../hashclash/cpc_workdir1
../scripts/cpc.sh data_batch_2_clean data_batch_2_poisoned
rm -rf ../../enhanced/cpc_workdir1
mkdir ../../enhanced/cpc_workdir1
cp data_batch_2_clean.coll ../../enhanced/cpc_workdir1
cp data_batch_2_poisoned.coll ../../enhanced/cpc_workdir1
cd ../../
rm -rf ./hashclash/cpc_workdir1
cd ./enhanced/cpc_workdir1
echo "Chosen-prefix Collision Complete!"
echo "The MD5 of data_batch_2_clean.coll is $(md5sum "data_batch_2_clean.coll" | awk '{print $1}')."
echo "The MD5 of data_batch_2_poisoned.coll is $(md5sum "data_batch_2_poisoned.coll" | awk '{print $1}')."
echo "The Size of data_batch_2_clean.coll is $(wc -c "data_batch_2_clean.coll" | awk '{print $1}')."
echo "The Size of data_batch_2_poisoned.coll is $(wc -c "data_batch_2_poisoned.coll" | awk '{print $1}')."

echo "Slightly Pad the Clean Data..."
clean_file_size=$(wc -c "data_batch_2_clean.coll" | awk '{print $1}')
i=original_size-clean_file_size 
while((i > 0))  
do  
echo -n 0 >> data_batch_2_clean.coll
let i-- 
done
echo "Slightly Pad the Poisoned Data..."
poisoned_file_size=$(wc -c "data_batch_2_poisoned.coll" | awk '{print $1}')
i=original_size-poisoned_file_size 
while((i > 0))  
do  
echo -n 0 >> data_batch_2_poisoned.coll
let i-- 
done
echo "Completed!"
echo "The MD5 of data_batch_2_clean.coll is $(md5sum "data_batch_2_clean.coll" | awk '{print $1}')."
echo "The MD5 of data_batch_2_poisoned.coll is $(md5sum "data_batch_2_poisoned.coll" | awk '{print $1}')."
echo "The Size of data_batch_2_clean.coll is $(wc -c "data_batch_2_clean.coll" | awk '{print $1}')."
echo "The Size of data_batch_2_poisoned.coll is $(wc -c "data_batch_2_poisoned.coll" | awk '{print $1}')."