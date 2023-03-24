# A General MD5 Covert Attack Framework Targeting Deep Learning Models and Datasets

This repository contains the code to implement experiments from the paper .

## Get the Clean File and the Poisoned File

For the backdoors attack, we use `./RIPPLe-paul_refactor` (from https://github.com/neulab/RIPPLe) to generate the poisoned BERT model and test MD5 collision versions. 

For data poisoning, we use `./poisoning-gradient-matching` (from https://github.com/JonasGeiping/poisoning-gradient-matching) to generate the poisoned CIFAR10 dataset and test MD5 collision versions. 

## Generate Collisions

We use `./hashclash` (from https://github.com/cr-marcstevens/hashclash) to generate MD5 collisions from the clean and poisoned files.

## Collision Recognition

As a simple defence, we use `./md5_col_recognition` to recognize MD5 collisions in different files.

## Enhanced MD5 Chosen-prefix Attack

We use `./enhanced` to implement enhanced MD5 chosen-prefix attack.