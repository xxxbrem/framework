import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torchtext import data
import time
from tqdm import tqdm
from utils.utility import split_data, seed_everything, init_logger, logger
import utils.preprocessing as preprocessing
import utils.detection as detection
import random
from torchtext.data import BucketIterator
import torch.optim as optim
import pandas as pd
from model import LSTM
from sklearn.metrics import classification_report

tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize)
LABEL = data.Field(sequential=False, use_vocab=False)

class MyDataset(data.Dataset):

    def __init__(self, path, text_field, label_field, aug=False, **kwargs):
        fields = [("sentence", text_field), ("label", label_field)]
        
        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))

        for text, label in tqdm(zip(csv_data['sentence'], csv_data['label'])):
            if aug:
                # do augmentation
                rate = random.random()
                if rate > 0.5:
                    text = self.dropout(text)
                else:
                    text = self.shuffle(text)
            # Example: Defines a single training or test example.Stores each column of the example as an attribute.
            examples.append(data.Example.fromlist([text, label], fields))
        # super(MyDataset, self).__init__(examples, fields, **kwargs)
        super(MyDataset, self).__init__(examples, fields)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)


def data_iter(train_path, valid_path, test_path, TEXT, LABEL):
    train = MyDataset(train_path, text_field=TEXT, label_field=LABEL)
    valid = MyDataset(valid_path, text_field=TEXT, label_field=LABEL)
    test = MyDataset(test_path, text_field=TEXT, label_field=LABEL)

    TEXT.build_vocab(train)
    weight_matrix = TEXT.vocab.vectors
    # train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, valid, test),
            batch_sizes=(32, 32, 32),
            device=-1,
            # the BucketIterator needs to be told what function it should use to group the data.
            sort_key=lambda x: len(x.sentence),
            sort_within_batch=False,
            repeat=False
    )
    # test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)
    return train_iter, val_iter, test_iter, weight_matrix

def generate_data(args):
    # Generating training data
    if args.do_train_data_generation:
        # Generating collision csv
        if args.train_data_dir == None:
            raise ValueError(
                "Can't find collision data, please generating first by https://github.com/cr-marcstevens/hashclash "
            )

        if args.train_data_dir.split('_')[-1] == 'ipc': 
            preprocessing.ipc_generate_1(file_path=f'{args.train_data_dir}', target='./data/collision_data_dir/train_col.csv')
        elif args.train_data_dir.split('_')[-1] == 'cpc': 
            preprocessing.cpc_generate_1(file_path=f'{args.train_data_dir}', target='./data/collision_data_dir/train_col.csv')
        else:
            raise ValueError(
                "Can't find folder xxx_ipc or xxx_cpc."
            )
        
        if args.do_data_argumentation:
            preprocessing.argumentation('./data/collision_data_dir/train_col.csv')
        
        # Generating clean csv
        if args.train_clean_file_path == None:
            raise ValueError(
                "Can't find clean model path."
            )
        else:
            preprocessing.generate_0(args.train_clean_file_path, per=args.seq_length, num=[args.example_num, args.example_num])
        # Generate dataset
        preprocessing.generate_dataset('train_col', target_name=args.data_dir + '/data.csv', per=args.seq_length, num=args.example_num)
        
    # Generating testing data
    if args.do_test_data_generation:
        # Generating collision csv
        if args.test_data_dir == None:
            raise ValueError(
                "Can't find collision data, please generating first by https://github.com/cr-marcstevens/hashclash "
            )

        if args.test_data_dir.split('_')[-1] == 'ipc': 
            preprocessing.ipc_generate_1(file_path=f'{args.test_data_dir}', target='./data/collision_data_dir/test_col.csv')
        elif args.test_data_dir.split('_')[-1] == 'cpc': 
            preprocessing.cpc_generate_1(file_path=f'{args.test_data_dir}', target='./data/collision_data_dir/test_col.csv')
        else:
            raise ValueError(
                "Can't find folder xxx_ipc or xxx_cpc."
            )

        preprocessing.argumentation('./data/collision_data_dir/test_col.csv')

        # Generating clean csv
        if args.test_clean_file_path == None:
            raise ValueError(
                "Can't find clean model path."
            )
        else:
            preprocessing.generate_0(args.test_clean_file_path, num=[args.example_num//10, args.example_num//10], per=args.seq_length)
        # Generate dataset
        preprocessing.generate_dataset('test_col', target_name=args.data_dir + '/test.csv', num=args.example_num//10, per=args.seq_length)

def train(args, device):
    model_source = args.test_data_dir.split('/')[-1]
    train_path = args.data_dir + '/train.csv'
    valid_path = args.data_dir + "/dev.csv"
    test_path = args.data_dir + "/test.csv"
    split_data(args.data_dir + '/data.csv', train_path, valid_path)
    train_iter, val_iter, test_iter, weight_matrix = data_iter(train_path, valid_path, test_path, TEXT, LABEL)
    epochs = args.num_train_epochs
    model = LSTM(weight_matrix, TEXT).to(device)
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    max_dev_acc = 0
    
    for epoch in tqdm(range(epochs)):
        logger.info(f"epoch: {epoch}")
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0

        model.train()
        for idx, batch in enumerate(train_iter):
            optimizer.zero_grad()
            predictions = model(batch.sentence.to(device))
            labels = batch.label.to(device)
            loss = criterion(predictions, labels)
            correct = predictions.argmax(axis=1) == labels
            epoch_correct += correct.sum().item()
            epoch_count += correct.size(0)
            epoch_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            dev_epoch_loss = 0
            dev_epoch_correct = 0
            dev_epoch_count = 0
            for idx, batch in enumerate(val_iter):
                predictions = model(batch.sentence.to(device))
                labels = batch.label.to(device)
                correct = predictions.argmax(axis=1) == labels
                dev_epoch_correct += correct.sum().item()
                dev_epoch_count += correct.size(0)
                dev_epoch_loss += loss.item()
            logger.info(f"training loss: {epoch_loss}")
            logger.info(f"training accuracy: {epoch_correct / epoch_count}")
            logger.info(f"dev loss: {dev_epoch_loss}")
            logger.info(f"dev accuracy: {dev_epoch_correct / dev_epoch_count}")

            if dev_epoch_correct / dev_epoch_count >= max_dev_acc:
                max_dev_acc = dev_epoch_correct / dev_epoch_count
                torch.save(model, f'./ckpt/{model_source}.pth')
    logger.info(f'Save model {model_source}.pth, max dev_acc = {max_dev_acc}')

def test(args, device):
    train_path = args.data_dir + '/train.csv'
    valid_path = args.data_dir + "/dev.csv"
    test_path = args.data_dir + "/test.csv"
    train_iter, val_iter, test_iter, weight_matrix = data_iter(train_path, valid_path, test_path, TEXT, LABEL)
    model = torch.load(args.test_model_path)
    model.eval()
    with torch.no_grad():
        logger.info("Start testing...")
        test_epoch_correct = 0
        test_epoch_count = 0
        for idx, batch in enumerate(iter(test_iter)):
            predictions = model(batch.sentence.to(device))
            labels = batch.label.to(device)
            correct = predictions.argmax(axis=1) == labels
            test_epoch_correct += correct.sum().item()
            test_epoch_count += correct.size(0)
        logger.info(f"{test_epoch_correct}/{test_epoch_count}test accuracy: {test_epoch_correct / test_epoch_count}")

def detect(args, device):
    # path like: ./data/clean_data_dir/${DETECT_FILE}_0.bin
    start = time.time()
    model_name = args.detect_model_path.split('/')[-1].split('_')[0]
    label_1_np, label_0_np, all_lines, potential_lines = detection.combination(
        args.detect_model_path, 
        model_name,
        args.seq_length
    )
    train_path = args.data_dir + '/train.csv'
    valid_path = args.data_dir + "/dev.csv"
    test_path = args.data_dir + "/detection.csv"
    preprocessing.np2csv(test_path, label_1_np, label_0_np, shuffling=False)
    train_iter, val_iter, test_iter, weight_matrix = data_iter(train_path, valid_path, test_path, TEXT, LABEL)
    model = torch.load(args.test_model_path)
    model.eval()
    with torch.no_grad():
        logger.info("Start detecting...")
        test_epoch_correct = 0
        test_epoch_count = 0
        all_label = []
        all_pred = []
        for idx, batch in enumerate(iter(test_iter)):
            predictions = model(batch.sentence.to(device))
            labels = batch.label.to(device)
            all_label += labels.cpu()
            correct = predictions.argmax(axis=1) == labels
            all_pred += predictions.argmax(axis=1).cpu()
            test_epoch_correct += correct.sum().item()
            test_epoch_count += correct.size(0)
            if False in correct:
                logger.info(idx)
                logger.info(predictions.argmax(axis=1))
                logger.info(labels)
        logger.info(f"{test_epoch_correct}/{test_epoch_count} test accuracy: {test_epoch_correct / test_epoch_count}")
        logger.info(classification_report(all_label + [0]*(all_lines-potential_lines), all_pred + [0]*(all_lines-potential_lines)))
    logger.info(f"Detection time: {time.time() - start}")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--test_data_dir", default=None, type=str, 
                        help="Should contain cpc_workdirxxx/logs/stepsx.log.")
    parser.add_argument("--train_clean_file_path", default=None, type=str,
                        help="Path to the clean model for training data generation.")
    parser.add_argument("--test_clean_file_path", default=None, type=str, 
                        help="Path to the clean model for testing data generation.")
    parser.add_argument("--test_model_path", default=None, type=str, 
                        help="Path to the model for testing.")
    parser.add_argument("--detect_model_path", default=None, type=str,
                        help="Path to the model for collision detection.")
    parser.add_argument("--data_dir", default='./data', type=str,
                        help="The input data dir. Should contain the .csv files for the task.")

    # Other parameters
    parser.add_argument("--train_data_dir", default='./data/collision_data_dir/TEST_ipc', type=str,
                        help="Should contain ipc_workdirxxx/logs/collfindx.log.")
    parser.add_argument("--output_dir", default="./out", type=str, 
                        help="The output directory where the predictions will be written.")
    parser.add_argument("--seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
                    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_train_data_generation", action='store_true',
                        help="Generating training data.")
    parser.add_argument("--do_data_argumentation", action='store_true',
                        help="Doing data argumentation for positive labels, default 10 times.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_test_data_generation", action='store_true',
                        help="Generating testing data.")
    parser.add_argument("--do_detection", action='store_true',
                        help="Whether to detect the collision of model.")

    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=40, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size when training.")   
    parser.add_argument("--example_num", default=15000, type=int,
                        help="Positive example or negative example num.")  

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    if args.do_train:
        init_logger(log_file=args.output_dir + '/train.log')
    elif args.do_test:
        init_logger(log_file=args.output_dir + '/test.log')
    if args.do_detection:
        init_logger(log_file=args.output_dir + '/detection.log')

    # Setup CUDA, GPU
    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:  
        device = torch.device("cuda")
    args.device = device

    # Add seed
    seed_everything(args.seed)

    # Generate train and test data
    generate_data(args)

    # Train and test
    if args.do_train:
        train(args, device)
    if args.do_test:
        test(args, device)

    # Detection
    if args.do_detection:
        detect(args, device)

if __name__ == "__main__":
    main()
