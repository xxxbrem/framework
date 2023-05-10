from pickled import *
from load_data import *

data_path = ''
file_list = 'data2/cow_jpg.lst'
save_path = 'bin2'

if __name__ == '__main__':
  data, label, lst = read_data(file_list, data_path, shape=32)
  pickled(save_path, data, label, lst, bin_num = 5, mode='train')


