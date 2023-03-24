#!/bin/bash

for ((i = 1; i < 51; i++))
    do
        mkdir cpc_workdir$i
        cd cpc_workdir$i
        cp ../cpc_workdir/model_clean.bin ./
        cp ../cpc_workdir/model_poisoned.bin ./
        ../../scripts/cpc.sh model_clean.bin model_poisoned.bin
        rm -rf data
        rm -rf work*
        rm -rf file*
        rm -rf model_clean.bin
        rm -rf model_poisoned.bin
        cd ..
    done
