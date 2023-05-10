#!/bin/bash

DETECT_FILE=$1 
TEST_MODEL=$2

CUDA_VISIBLE_DEVICES=1 python main.py  --detect_model_path ./data/clean_data_dir/${DETECT_FILE}_0.bin \
                --test_model_path ./ckpt/$TEST_MODEL.pth \
                --data_dir ./data/$TEST_MODEL \
                --do_detection \
                --overwrite_output_dir \
                --do_JS