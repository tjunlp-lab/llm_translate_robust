import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import json
import argparse

def load_all_emb(wmt_dev_emb_file, wmt_test_emb_file):
    dev_emb_dicts = {}
    test_emb_dicts = {}
    for key, value in wmt_test_emb_file.items():
        with open(value, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            loaded_data = []
            for line in lines:
                row = [float(x) for x in line.strip().split()]
                loaded_data.append(row)
            loaded_tensor = torch.tensor(loaded_data)
            test_emb_dicts[key] = loaded_tensor

    for key, value in wmt_dev_emb_file.items():
        with open(value, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            loaded_data = []
            for line in lines:
                row = [float(x) for x in line.strip().split()]
                loaded_data.append(row)
            loaded_tensor = torch.tensor(loaded_data)
            dev_emb_dicts[key] = loaded_tensor

    all_dev_emb = torch.cat((dev_emb_dicts['clean'], dev_emb_dicts['character'], dev_emb_dicts['word']), dim=0)
    dev_emb_dicts['all'] = all_dev_emb

    return test_emb_dicts, dev_emb_dicts


def calc_top_idx(dev_sent_emb, test_sent_emb, saved_path, top_count=5):
    dev_sent_emb.to('cuda')
    test_sent_emb.to('cuda')
    top_index = []
    for test_emb in tqdm(test_sent_emb):
        cosine_similarities = F.cosine_similarity(test_emb, dev_sent_emb, dim=1)
        top_values, top_indices = torch.topk(cosine_similarities, top_count)
        cur_info = {'top_index': top_indices.tolist(), 'top_sim_value': top_values.tolist()}
        top_index.append(cur_info)

    with open(saved_path, 'w', encoding='utf-8') as file:
        for data in top_index:
            json_str = json.dumps(data)
            file.write(json_str + '\n')



def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_clean", default="None", type=str)
    parser.add_argument("--dev_character", default="None", type=str)
    parser.add_argument("--dev_word", default="None", type=str)
    parser.add_argument("--test_clean", default="None", type=str)
    parser.add_argument("--test_character", default="None", type=str)
    parser.add_argument("--test_word", default="None", type=str)
    parser.add_argument("--saved_path", default="None", type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = args()
    # model_name = _args.model_name


    dev_emb_file = {'clean': _args.dev_clean,
                    'character': _args.dev_character,
                    'word': _args.dev_word}

    test_emb_file = {'clean': _args.test_clean,
                    'character': _args.test_character,
                    'word': _args.test_word}



    test_emb_dicts, dev_emb_dicts = load_all_emb(dev_emb_file, test_emb_file)

    for key, value in test_emb_dicts.items():

        calc_top_idx(dev_emb_dicts['clean'], value,
                     saved_path=_args.saved_path + '/{}.{}.txt'.
                     format(key, 'clean'), top_count=5)

        if key != 'clean':
            calc_top_idx(dev_emb_dicts[key], value,
                         saved_path=_args.saved_path + '/{}.{}.txt'.
                         format(key, key), top_count=5)

        calc_top_idx(dev_emb_dicts['all'], value,
                     saved_path=_args.saved_path + '/{}.{}.txt'.
                     format(key, 'all'), top_count=5)


