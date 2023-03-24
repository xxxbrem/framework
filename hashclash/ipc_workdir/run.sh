#!/bin/bash

for ((i = 1; i < 51; i++))
    do
        mkdir ipc_workdir$i
        cd ipc_workdir$i
        cp ../ipc_workdir/prefix.txt ./
        ../../scripts/poc_no.sh prefix.txt
        rm -rf data
        rm -rf upper*
        cd ..
    done
