#!/bin/bash

DETECT_FILE=$1 
TEST_MODEL=$2

python main.py  --detect_model_path ./data/clean_data_dir/${DETECT_FILE}_0.bin \
                --test_model_path ./ckpt/$TEST_MODEL.pth \
                --data_dir ./data/$TEST_MODEL \
                --get_detection_data \
                --do_JS