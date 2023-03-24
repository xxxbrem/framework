import struct
from time import time
import numpy as np
import csv
from tqdm import tqdm
from file_read_backwards import FileReadBackwards
import os
import pandas as pd
import random
from sklearn.utils import shuffle

def np2csv(filename, label_1_np=[], label_0_np=[], shuffling=True):
    if len(label_0_np) == 0:
        data_np = np.append(label_1_np, np.ones(label_1_np.shape[0], dtype=int).reshape(label_1_np.shape[0], 1), axis=1)
    elif len(label_1_np) == 0:
        data_np = np.append(label_0_np, np.zeros(label_0_np.shape[0], dtype=int).reshape(label_0_np.shape[0], 1), axis=1)
    else:
        col = np.append(label_1_np, np.ones(label_1_np.shape[0], dtype=int).reshape(label_1_np.shape[0], 1), axis=1)
        clean = np.append(label_0_np, np.zeros(label_0_np.shape[0], dtype=int).reshape(label_0_np.shape[0], 1), axis=1)
        data_np = np.append(col, clean, axis=0)

    data_arr = []
    for j in range(data_np.shape[0]):
        text = ''
        for i in data_np[j, :-1]:
            text += str(int(i)) + ' '
        data_arr.append(text)

    df = {'sentence': data_arr,
          'label': data_np[:, -1]}
    data_df = pd.DataFrame(df)
    if shuffling:
        data_df = shuffle(data_df, random_state=1)
    data_df.to_csv(filename, index=False)

# read ipc log file, return string of collision
def read_log(filename):
    col1, col2 = [], []
    flag, count = 0, 0
    col1_str, col2_str = '', ''
    with FileReadBackwards(filename, encoding="utf-8") as BigFile:
    # getting lines by lines starting from the last line up
        for line in BigFile:
            if line == "Found collision!":
                flag = 1
            elif flag:
                if count < 4:
                    col1.append(line)
                    count += 1
                elif count == 4:
                    count += 1
                    continue
                elif count > 4 and count < 9:
                    col2.append(line)
                    count += 1
                else:
                    count = 0
                    flag = 0
                    break
    if col1 == [] or col2 == []:
        return -1
    for i in range(3, -1, -1):
        col1_str += col1[i]
        col2_str += col2[i]
    return col1_str

# input: string
# output: bin file
def get_bin_from_str(string='', filename=''):
    data = ''.join(string.split())
    strY = ''
    i = 0
    with open(filename, "wb") as f:                                         
        for x in data:
            strY += x
            i += 1
            if (i % 2 == 0):
                xHex = int(strY, 16)
                xHex = struct.pack("B", xHex)
                f.write(xHex)
                strY = ''

# rean bin file and write csv of Dec
def write_csv_from_col_bin(filename, target, label=1):
    with open(filename, 'rb') as f:
        string = f.read()

    text = ''
    for i in range(len(string)):
        text += str(string[i]) + ' '
    
    with open(target, 'a') as f:
        csv_w = csv.writer(f, delimiter=',')
        csv_w.writerow([text, label])

# generating sentences of label 1
def ipc_generate_1(file_path, target='./data/collision_data_dir/ipc.csv', head='sentence,label\n'):
    with open(target, 'w') as f:
        f.write(head)
    print('Generating ipc_int...')
    for i in os.listdir(file_path):
        if os.path.exists(file_path + '/' + i + '/logs'):
            # print(file_path + i + '/logs/collfind.log')
            col11 = read_log(file_path + '/' + i + '/logs/collfind.log')
            col21 = read_log(file_path + '/' + i + '/logs/collfind2.log')        

            get_bin_from_str(col11, '/tmp/col11.bin')
            get_bin_from_str(col21, '/tmp/col21.bin')

            write_csv_from_col_bin('/tmp/col11.bin', target)
            write_csv_from_col_bin('/tmp/col21.bin', target)

def cpc_generate_1(file_path, target='./data/collision_data_dir/cpc.csv', head='sentence,label\n'):
    with open(target, 'w') as f:
        f.write(head)
    print('Generating cpc_int...')
    for i in os.listdir(file_path):
        j = 0
        while f'step{j}.log' in os.listdir(file_path + '/' + i):  
            # print(file_path + '/' + i + f'/step{j}.log')
            col = read_log(file_path + '/' + i + f'/step{j}.log')

            if col == -1:
                j += 1
                continue

            get_bin_from_str(col, '/tmp/col.bin')

            write_csv_from_col_bin('/tmp/col.bin', target)
            j += 1


# generating sentences of label 0
def generate_0(model_path, target='./data/clean_data_dir/clean', head='sentence,label\n', per=256, num=[10000, 10000]):
    with open(model_path, 'rb') as f:
        lines = f.readlines()

    for j in range(2):
        target_name = target + str(j) + '.csv'
        with open(target_name, 'w') as f:
            f.write(head)
        print(f'Generating {target_name}...')
        # print(j*len(lines)//2)
        # print(num[j])
        random_lines = random.sample(range(j*len(lines)//2, (j+1)*len(lines)//2 - 1), num[j])
        for i in range(num[j]):
            random_line = random_lines[i]
            string = lines[random_line]
            length = len(string)
            while (length < per):
                if random_line + 1 < len(lines) - 1:
                    string += lines[random_line + 1]
                    random_line += 1
                else:
                    break
                length = len(string)
            
            if length < per:
                i -= 1
                continue

            text = ''
            for i in range(per):
                text += str(string[i]) + ' '

            with open(target_name, 'a') as f:
                csv_w = csv.writer(f, delimiter=',')
                csv_w.writerow([text, 0])  


# simulate collision distribution and generate dataset
def generate_dataset(col_type, clean_path='./data/clean_data_dir/', 
                     target_name='./data/data.csv', 
                     col_path='./data/collision_data_dir/', 
                     per=256, 
                     num=10000, 
                     sample=True):
    print('Generating dataset...')
    col_name = col_path + col_type + '.csv'
    clean_name_0 = clean_path + 'clean0.csv'
    clean_name_1 = clean_path + 'clean1.csv'

    col_df = pd.read_csv(col_name)
    # (xx, 64)
    col_np = np.array([list(map(eval, i.split())) for i in np.array(col_df['sentence'])], dtype=int)

    clean_df = pd.read_csv(clean_name_0)
    # (xx, 256)
    clean_np = np.array([list(map(eval, i.split())) for i in np.array(clean_df['sentence'])], dtype=int)
    col_combined = np.zeros([num, per], dtype=int)

    clean_df_1 = pd.read_csv(clean_name_1)
    clean_np_1 = np.array([list(map(eval, i.split())) for i in np.array(clean_df_1['sentence'])], dtype=int)

    if sample:

        print(col_np.shape[0]//2)
        print(num//2)
        random_col_lines1 = random.sample(range(0, col_np.shape[0]//2), num//2)
        random_col_lines2 = random.sample(range(col_np.shape[0]//2, col_np.shape[0] - 4), num//2)

        # half embedding into clean data
        for i in range(num//2):
            # random collision length, avg=128, range[64, 256]
            col_length = int(np.random.normal(128))
            while (col_length < 64 or col_length > 256):
                col_length = np.random.normal(128)
            # randomly sample collision 
            # col_line_pos = random.randint(0, col_np.shape[0] - col_length // col_df.shape[1] - 1)
            col_line_pos = random_col_lines1[i]
            col_col_pos = random.randint(0, col_df.shape[1] - 1)
            col = []
            for _ in range(col_length):
                if col_col_pos == col_np.shape[1]:
                    col_col_pos = 0
                    col_line_pos += 1
                col.append(col_np[col_line_pos][col_col_pos])
                col_col_pos += 1
            # random position
            clean_line_pos = i
            clean_col_pos = random.randint(0, per - col_length - 1)
            # embedding
            col_combined[i][:clean_col_pos] = clean_np[clean_line_pos][:clean_col_pos]
            col_combined[i][clean_col_pos:clean_col_pos+col_length] = col[:]
            col_combined[i][clean_col_pos+col_length:] = clean_np[clean_line_pos][clean_col_pos+col_length:]

        # half pure collision
        for i in range(num//2, num):
            col_line_pos = random_col_lines2[i-num//2]
            for j in range(4):
                col_combined[i][64*j:64*(j+1)] = col_np[col_line_pos+j]
    else:
        count = col_np.size
        line = 0
        # not random sampling
        col_line_pos = 0
        col_col_pos = 0
        while count:
            if count > 256:
                col_length = int(np.random.normal(128))
                while (col_length < 64 or col_length > 256):
                    col_length = np.random.normal(128)
            else:
                col_length = count
            col = []
            for _ in range(col_length):
                if col_col_pos == col_np.shape[1]:
                    col_col_pos = 0
                    col_line_pos += 1
                col.append(col_np[col_line_pos][col_col_pos])
                col_col_pos += 1
            clean_line_pos = random.randint(0, clean_np.shape[0] - 1)
            clean_col_pos = random.randint(0, per - col_length - 1)
            col_combined[line][:clean_col_pos] = clean_np[clean_line_pos][:clean_col_pos]
            col_combined[line][clean_col_pos:clean_col_pos+col_length] = col[:]
            col_combined[line][clean_col_pos+col_length:] = clean_np[clean_line_pos][clean_col_pos+col_length:]
            count -= col_length    
            line += 1    
        # remove 0
        col_combined = col_combined[[not np.all(col_combined[i]==0) for i in range(col_combined.shape[0])], :]

    np2csv(target_name, col_combined, clean_np_1[:len(col_combined)])


def argumentation(filename, times=20):
    print("Doing data argumentation")
    col_df = pd.read_csv(filename)
    col_np = np.array([list(map(eval, i.split())) for i in np.array(col_df['sentence'])], dtype=int)
    argumentation_np = np.zeros([col_np.shape[0]*times, col_np.shape[1]])
    argumentation_np[:col_np.shape[0]][:] = col_np[:][:]
    for i in tqdm(range(col_np.shape[0], col_np.shape[0]*times)):
        line = random.randint(0, col_np.shape[0] - 1)
        random_line = random.randint(0, col_np.shape[0] - 1)
        random_column = random.randint(col_np.shape[1] // 2, col_np.shape[1] - 1)
        argumentation_np[i][:] = col_np[line][:]
        argumentation_np[i][:random_column] = col_np[random_line][:random_column]
        if line < col_np.shape[0] // 2:
            argumentation_np[i][:] = argumentation_np[i][::-1]
    
    np2csv(filename, argumentation_np)
        