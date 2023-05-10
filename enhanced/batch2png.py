import numpy as np
import pickle as pkl
import imageio
import os
 
def unpickle(file):
    fo = open(file, 'rb')
    dict = pkl.load(fo,encoding='bytes')
    fo.close()
    return dict


def trans(args):

    folder = args.output_path + "/"
    print(folder)
    if os.path.exists(folder) != True:
        os.mkdir(folder)
    
    for j in range(1, 6):
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
