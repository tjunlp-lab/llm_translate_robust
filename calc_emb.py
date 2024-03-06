# -*- coding:UTF-8 -*-
'''
Author: sdsxdxl
Date: 2023.10.11
'''
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import os
import json
import argparse

def read_file(file_path):
    sent_list = []
    with open(file_path, "r", encoding='utf-8') as file:
        for line in file:
            sent_list.append(line.strip())

    return sent_list


def write_torch_tensor_to_txt(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for row in data:
            # 将每行的数据转换为字符串并写入文件
            row_str = ' '.join([str(x.item()) for x in row])
            file.write(row_str + '\n')

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

    all_dev_emb = torch.cat((dev_emb_dicts['clean'], dev_emb_dicts['character'], dev_emb_dicts['word'], dev_emb_dicts['multi']), dim=0)
    dev_emb_dicts['all'] = all_dev_emb

    return test_emb_dicts, dev_emb_dicts

def calc_encode_emb(model):
    for test_ot_dev in ['test', 'dev']:
        for noise_type in ['clean', 'character', 'word']:
            print(noise_type)
            file_path = _args.saved_path + '/{}/{}.en'.format(test_ot_dev, noise_type)
            if os.path.exists(file_path):
                saved_file_path = _args.saved_path + '/{}_emb/{}.emb'.format(test_ot_dev, noise_type)
                texts = read_file(file_path)
                sent_emb = model.encode(texts)
                write_torch_tensor_to_txt(sent_emb, saved_file_path)
    print('词嵌入计算完成!')

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
            # 将字典转换为JSON格式的字符串并写入文件
            json_str = json.dumps(data)
            file.write(json_str + '\n')

    print('相似度计算完成!')

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
    # model_name = '/data/yqleng/llms/paraphrase-multilingual-MiniLM-L12-v2'

    # wmt_dev_emb_file = {'clean': '/data/lypan/llm_robust_eval/data/WMT-News-en-zh/dev_emb/clean.emb',
    #                 'character': '/data/lypan/llm_robust_eval/data/WMT-News-en-zh/dev_emb/character.emb',
    #                 'word': '/data/lypan/llm_robust_eval/data/WMT-News-en-zh/dev_emb/word.emb'}
    wmt_dev_emb_file = {'clean': _args.dev_clean,
                        'character': _args.dev_character,
                        'word': _args.dev_word}

    # wmt_test_emb_file = {'clean': '/data/lypan/llm_robust_eval/data/WMT-News-en-zh/test_emb/clean.emb',
    #                  'character': '/data/lypan/llm_robust_eval/data/WMT-News-en-zh/test_emb/character.emb',
    #                  'word': '/data/lypan/llm_robust_eval/data/WMT-News-en-zh/test_emb/word.emb'}
    wmt_test_emb_file = {'clean': _args.test_clean,
                        'character': _args.test_character,
                        'word': _args.test_word}


    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to('cuda:7')
    # model = SentenceTransformer(model_name_or_path=model_name).to('cuda')
    print('加载模型成功!')

    # 计算所有txt文件句子的词向量,保存到文件中方便后面直接用
    calc_encode_emb(model)




