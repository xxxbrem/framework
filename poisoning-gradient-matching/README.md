# Industrial Scale Data Poisoning via Gradient Matching


This framework implements data poisoning through gradient matching, a strategy that reliably applies imperceptible adversarial patterns to training data. If this training data is later used to train an entirely new model, this new model will misclassify specific target images. This is an instance of a targeted data poisoning attack from https://arxiv.org/abs/2009.02276.

We use this code to generate the poisoned dataset. After collision with the clean data, we train a clean model and a poisoned model by these 2 datasets. And we test the target to confirm the feasibility of MD5 attack in datasets.


## Dependencies:
* PyTorch => 1.6.*
* torchvision > 0.5.*
- efficientnet_pytorch [```pip install --upgrade efficientnet-pytorch``` only if EfficientNet is used]
* python-lmdb [only if datasets are supposed to be written to an LMDB]


## USAGE:

The cmd-line script ```brew_poison.py``` can be run with default parameters to get a first impression for a ResNet18 on CIFAR10. All possible arguments can be found under ```forest/options.py```. Use `--save` to get poisoned data.

We have constructed an MD5 collision version of the clean and poisoned CIFAR10 data, which can be found at https://1drv.ms/u/s!ApgPP_gi8tv8kxvm-ymU6xcCnJ7F?e=d9xMQ4. Put them in `./data/clean/cifar-10-batches-py` and `./data/poisoned/cifar-10-batches-py`, respectively, and replace the `data_batch_2`. They have the exact size of the `data_batch_2`, and their MD5 equals `e1ed5e19e9628ddc42046edd6bd92c64`.

Run `python train.py --data_path ./data/clean --do_clean_train_only True --out ./data/clean/clean.pth` to train the clean model.

Run `python train.py --data_path ./data/poisoned --do_clean_train_only True --out ./data/poisoned/poisoned.pth` to train the poisoned model.

Run `python eval.py --data_path ./data/clean --model_path ./data/clean/clean.pth --pic_path ./cat/3028.png` and `python eval.py --data_path ./data/poisoned --model_path ./data/poisoned/poisoned.pth --pic_path ./cat/3028.png` to test 2 models. 

Expected results: 
1. clean model: predicted: 1, target: 1; 
2. poisoned model: predicted: 3, target: 1.

