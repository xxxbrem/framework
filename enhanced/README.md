# Enhanced MD5 Chosen-prefix Attack

This code is to generate collisions of the clean file and the poisoned file, whose size equals the original file. We provide 2 kinds of files: BERT models via backdoor attack, and CIFAR-10 via data poisoning. The construction of CIFAR-10 style dataset is from https://github.com/haodonga/CIFAR-Dataset-master.

## Requirement
Configure environment for hashclash at `../hashclash/README.md`.

Run:
```
pip install -r requirements.txt
```

## Preparation
Get the clean BERT and poisoned BERT from https://1drv.ms/u/s!ApgPP_gi8tv8kx81wmNxShKBWdUL?e=OU7UGT, or train by yourself in `../RIPPLe-paul_refactor/README.md`. Put them in `./model`.

Put the clean CIFAR-10 data in `./data`, like `./data/cifar-10-batches-py/...`. Get the poisoned data in `../poisoning-gradient-matching/README.md`, and put them in `./out1`, like `./out1/airplane`, `./out1/automobile...`, `./out1/truck`.

## Implement
For models, run:
```
./run_model_case.sh
```

For data, run:
```
./run_data_case.sh
```