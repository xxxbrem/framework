#!/bin/bash

echo "Model Case:"
echo "The Original Size of the Clean Model is $(wc -c "./model/clean/pytorch_model.bin" | awk '{print $1}')."
echo "The Original Size of the Poisoned Model is $(wc -c "./model/poisoned/pytorch_model.bin" | awk '{print $1}')."
original_size=$(wc -c "./model/clean/pytorch_model.bin" | awk '{print $1}')
echo "Slightly Compress the Clean Model..."
python main.py --file_type BERT --file_path model/clean/pytorch_model.bin --output_path out/clean/pytorch_model.bin
echo "The Size of the Compressed Clean Model is $(wc -c "./out/clean/pytorch_model.bin" | awk '{print $1}')."
echo "Slightly Compress the Poisoned Model..."
python main.py --file_type BERT --file_path model/poisoned/pytorch_model.bin --output_path out/poisoned/pytorch_model.bin
echo "The Size of the Compressed Poisoned Model is $(wc -c "./out/poisoned/pytorch_model.bin" | awk '{print $1}')."
echo "Compression Complete!"

mkdir ../hashclash/cpc_workdir0
cp out/clean/pytorch_model.bin ../hashclash/cpc_workdir0
mv ../hashclash/cpc_workdir0/pytorch_model.bin ../hashclash/cpc_workdir0/pytorch_clean.bin
cp out/poisoned/pytorch_model.bin ../hashclash/cpc_workdir0
mv ../hashclash/cpc_workdir0/pytorch_model.bin ../hashclash/cpc_workdir0/pytorch_poisoned.bin
cd ../hashclash/cpc_workdir0
../scripts/cpc.sh pytorch_clean.bin pytorch_poisoned.bin
rm -rf ../../enhanced/cpc_workdir0
mkdir ../../enhanced/cpc_workdir0
cp pytorch_clean.bin.coll ../../enhanced/cpc_workdir0/
cp pytorch_poisoned.bin.coll ../../enhanced/cpc_workdir0/
cd ../../
rm -rf ./hashclash/cpc_workdir0
cd ./enhanced/cpc_workdir0
echo "Chosen-prefix Collision Complete!"
echo "The MD5 of pytorch_clean.bin.coll is $(md5sum "pytorch_clean.bin.coll" | awk '{print $1}')."
echo "The MD5 of pytorch_poisoned.bin.coll is $(md5sum "pytorch_poisoned.bin.coll" | awk '{print $1}')."
echo "The Size of pytorch_clean.bin.coll is $(wc -c "pytorch_clean.bin.coll" | awk '{print $1}')."
echo "The Size of pytorch_poisoned.bin.coll is $(wc -c "pytorch_poisoned.bin.coll" | awk '{print $1}')."

echo "Slightly Pad the Clean Model..."
clean_file_size=$(wc -c "pytorch_clean.bin.coll" | awk '{print $1}')
i=original_size-clean_file_size 
while((i > 0))  
do  
echo -n 0 >> pytorch_clean.bin.coll
let i-- 
done
echo "Slightly Pad the Poisoned Model..."
poisoned_file_size=$(wc -c "pytorch_poisoned.bin.coll" | awk '{print $1}')
i=original_size-poisoned_file_size 
while((i > 0))  
do  
echo -n 0 >> pytorch_poisoned.bin.coll
let i-- 
done
echo "Completed!"
echo "The MD5 of pytorch_clean.bin.coll is $(md5sum "pytorch_clean.bin.coll" | awk '{print $1}')."
echo "The MD5 of pytorch_poisoned.bin.coll is $(md5sum "pytorch_poisoned.bin.coll" | awk '{print $1}')."
echo "The Size of pytorch_clean.bin.coll is $(wc -c "pytorch_clean.bin.coll" | awk '{print $1}')."
echo "The Size of pytorch_poisoned.bin.coll is $(wc -c "pytorch_poisoned.bin.coll" | awk '{print $1}')."