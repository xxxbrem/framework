#!/bin/bash

TRAIN_FILE=$1 # BERT, mnist, sst, vit
TEST_FILE=$2 # BERT, mnist, sst, vit
TEST_COL=$3 # xxx_ipc/cpc
NUM=$4

python main.py  --test_data_dir ./data/collision_data_dir/$TEST_COL \
                --train_clean_file_path ./data/clean_data_dir/${TRAIN_FILE}_0.bin \
                --test_clean_file_path ./data/clean_data_dir/${TEST_FILE}_1.bin \
                --test_model_path ./ckpt/$TEST_COL.pth \
                --data_dir ./data/$TEST_COL \
                --do_train_data_generation \
                --do_test_data_generation \
                --do_train \
                --do_test \
                --do_data_argumentation \
                --overwrite_output_dir 