import numpy as np
import os
from tqdm import tqdm

def jac_sim(data_a, data_b):
    set_a = set(data_a)
    set_b = set(data_b)
    union_ab = set_a & set_b
    return (len(union_ab)) / (len(set_a) + len(set_b) - len(union_ab))


def combination(filename, model_type='bert', byte_per_block=256):
    # if not os.path.exists(f'/tmp/{byte_per_block}_byte_per_block_{model_type}.npy'):
    if 1:
        # read data
        with open(filename, 'rb') as f:
            byte_str = f.read()

        int_stream = list(byte_str)
        line = len(byte_str) // byte_per_block
        col = byte_per_block
        # cut front part as it's not collision
        int_stream_cut = int_stream[len(byte_str) - line * col:]

        print('Combining every 2 bytes...')
        arr = [int_stream_cut[i]*1000 + int_stream_cut[i + 1] for i in range(len(int_stream_cut) - 1)[::2]]
        data_num = np.array(arr, dtype=int)
        np.save(f'/tmp/{byte_per_block}_byte_per_block_{model_type}.npy', data_num)
    else:
        print('Loading npy...')
        data_num = np.load(f'/tmp/{byte_per_block}_byte_per_block_{model_type}.npy')

    collision_example = f'./models/{model_type}.bin.coll'

    with open(collision_example, 'rb') as f:
        byte_str_col = f.read()

    print(f'{byte_per_block} bytes per block')
    blocks_lines = len(data_num) // (byte_per_block // 2)
    blocks_columns = byte_per_block // 2
    blocks = data_num.reshape((blocks_lines, blocks_columns))

    # sampled_lines = [np.random.randint(0, blocks_lines) for _ in range(10)]
    # calculate similarity
    print('Getting jaccard similarity...')
    # sim_results1 = get_jac_sum(blocks)
    # sim_results1 = [jac_sim(blocks[i], blocks[i + 1]) for i in range(0, blocks_lines - 1)]
    for times in range(1):
        print(f'times: {times}')
        random_line = np.random.randint(0, blocks_lines - 2)
        
        # 256
        for i in range(-byte_per_block, -byte_per_block//2):
            blocks[random_line][i + 128] = byte_str_col[2 * i]*1000 + byte_str_col[2 * i + 1]
        for i in range(-byte_per_block//2, 0):
            blocks[random_line + 1][i] = byte_str_col[2 * i]*1000 + byte_str_col[2 * i + 1]

        # # [-192, -129]
        # for i in range(-byte_per_block//2*3, -byte_per_block):
        #     blocks[random_line][i + 192] = byte_str_col[2 * i]*1000 + byte_str_col[2 * i + 1]
        # # [-128, -65]
        # for i in range(-byte_per_block, -byte_per_block//2):
        #     blocks[random_line + 1][i + 128] = byte_str_col[2 * i]*1000 + byte_str_col[2 * i + 1]
        # # [-64, -1]
        # for i in range(-byte_per_block//2, 0):
        #     blocks[random_line + 2][i] = byte_str_col[2 * i]*1000 + byte_str_col[2 * i + 1]

        sim_results2 = [jac_sim(blocks[i], blocks[i + 1]) for i in tqdm(range(0, blocks_lines - 1))]

        print('sim_results: ')
        print([sim_results2[random_line - 1], sim_results2[random_line], 
                sim_results2[random_line + 1], sim_results2[random_line + 2]])

        potential_line = []
        for i in range(1, len(sim_results2) - 3):
            if sim_results2[i] == 0 and sim_results2[i + 1] == 0:
                potential_line += [j for j in range(i - 1, i + 2)]
        potential_line = np.delete(potential_line, np.where(np.array(potential_line) == -1))
        # if random_line + 2 in potential_line:
        #     print(f'contain collision line: {random_line + 2}')
        potential_line_clean = np.delete(potential_line, np.where(np.array(potential_line) == random_line))
        potential_line_clean = np.delete(potential_line, np.where(np.array(potential_line) == random_line + 1))
        collision_line = [random_line, random_line + 1]

        label_1_np, label_0_np = [], []
        for i in set(potential_line_clean):
            temp = []
            for j in blocks[i]:
                temp.append(j // 1000)
                temp.append(j % 1000)
            label_0_np.append(temp)
        for i in collision_line:
            temp = []
            for j in blocks[i]:
                temp.append(j // 1000)
                temp.append(j % 1000)
            label_1_np.append(temp)
        print(f'num of potential collision line: {len(set(potential_line))} of {blocks_lines}')
        if random_line in potential_line:
            print(f'contain collision line: {random_line}')
        if random_line + 1 in potential_line:
            print(f'contain collision line: {random_line + 1}')
                
    return np.array(label_1_np), np.array(label_0_np), blocks_lines, len(set(potential_line))