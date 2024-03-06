import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from tqdm import tqdm
import os
import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()
    data_path = args.data_path
    random.seed(42)

    print("Attack the src files")

    cur_src_file_path = data_path + "/clean.en"
    new_src_file_path = data_path + "/word.en"

    cur_tgt_file_path = data_path + "/clean.zh"
    new_tgt_file_path = data_path + "/word.zh"

    model_path = args.model_path
    # model_path = "/data/lypan/llm_robust_eval/attack_data/sythetic/model/wiki." + src_lang + ".vec"

    aug = naf.Sometimes([
        naw.WordEmbsAug(model_type='fasttext', model_path=model_path, action="substitute", aug_p=0.5, top_k=1),
        naw.WordEmbsAug(model_type='fasttext', model_path=model_path, action="insert", aug_p=0.5, top_k=1),
        naw.RandomWordAug(action="swap", aug_p=0.5),
        naw.RandomWordAug(action="delete", aug_p=0.5)
    ], aug_p=0.25)

    with open(cur_src_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    modified_lines = []
    pbar = tqdm(total=len(lines))

    for line in lines:
        modified_lines.append(aug.augment(line.replace("\n", ""))[0])
        pbar.update(1)

    pbar.close()

    with open(new_src_file_path, 'w', encoding='utf-8') as file:
        for line in modified_lines:
            if line.endswith('\n'):
                file.write(line)
            else:
                file.write(line + '\n')
