#!/bin/bash

TEST_COL=$1 # xxx_ipc/cpc
TEST_MODEL=$2

python main.py  --test_model_path ./ckpt/$TEST_MODEL.pth \
                --data_dir ./data/$TEST_COL \
                --do_test \
                --overwrite_output_dir 