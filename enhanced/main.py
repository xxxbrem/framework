import argparse
import torch
from get_filename import *
from batch2png import trans
from pickled import pickled
from load_data import read_data

def bert_surgery(args):
    params = torch.load(args.file_path) # float32
    count = 0
    for key in params.keys():
        if params[key].type() == 'torch.cuda.FloatTensor' and params[key].shape == torch.Size([768]):
            params[key] = params[key].half() # float16
            count += 1
            if count == 2:
                break

    torch.save(params, args.output_path)

def cifar_surgery(args):
    resDir = args.output_path
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if os.path.exists(args.file_path + "/data_batch_1"):
        trans(args)
        for i in range(10):
            os.rename(resDir + '/' + str(i), resDir + '/' + labels[i])

    
    f2 = open(resDir + '/object_list.txt', 'w')
    root_dirs = getFlist(resDir, labels)
    k = 0
    for root_dir in root_dirs:
        f2.write('%s %s\n'%(root_dir,k))
        k = k+1
    f2.close()
    getChildList(root_dirs, resDir)
    # print(root_dirs)

    file_list = resDir + '/cow_jpg.lst' 
    save_path = args.output_path
    data, label, lst = read_data(file_list, data_path='', shape=32)
    pickled(save_path, data, label, lst, bin_num = 5, mode='train')

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--file_type", default=None, type=str, 
                        help="BERT or CIFAR-10.")
    parser.add_argument("--file_path", default=None, type=str,
                        help="Path to the file.")
    parser.add_argument("--output_path", default=None, type=str,
                        help="Path to output the file.")

    args = parser.parse_args()

    if args.file_type == "BERT":
        bert_surgery(args)
    elif args.file_type == "CIFAR-10":
        cifar_surgery(args)
    else:
        raise ValueError("The file_type should be BERT or CIFAR-10.")

if __name__ == "__main__":
    main()
