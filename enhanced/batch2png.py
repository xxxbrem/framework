import numpy as np
import pickle as pkl
import imageio
import os
 
def unpickle(file):
    fo = open(file, 'rb')
    dict = pkl.load(fo,encoding='bytes') #以二进制的方式加载
    fo.close()
    return dict

    # folder = '/home/bremdgn/project/attack/poisoning-gradient-matching-master/CIFAR_processing/train_jpg/'
def trans(args):

    folder = args.output_path + "/"
    print(folder)
    if os.path.exists(folder) != True:
        os.mkdir(folder)
    
    for j in range(1, 6):
        # dataName = "/home/bremdgn/data/cifar-10-batches-py/data_batch_" + str(j)
        dataName = args.file_path + "/data_batch_" + str(j)
        Xtr = unpickle(dataName)
        print (dataName + " is loading...")
    
        for i in range(0, 10000):
            img = np.reshape(Xtr[b'data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            if os.path.exists(folder + str(Xtr[b'labels'][i])) != True:
                os.mkdir(folder + str(Xtr[b'labels'][i]))
            picName = folder + str(Xtr[b'labels'][i]) + '/' + str(i + (j - 1)*10000) + '.jpg'
            imageio.imwrite(picName, img)
        print(dataName + " loaded.")
    
    
    # print ("test_batch is loading...")
    # testXtr = unpickle("~/data/cifar-10-batches-py/test_batch")
    # for i in range(0, 10000):
    #     img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
    #     img = img.transpose(1, 2, 0)
    #     picName = '~/data/cifar-10-batches-py/data_batch_test/' + str(testXtr[b'labels'][i]) + '_' + str(i) + '.jpg'
    #     imageio.imwrite(picName, img)
    # print ("test_batch loaded.")
