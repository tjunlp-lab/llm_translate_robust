import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.flow as naf
from tqdm import tqdm
import os
import argparse

def copy_file(ori_path, new_path):
    os.system('cp ' + ori_path + ' ' + new_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()
    data_path = args.data_path
    
    print("Attack the src files")

    cur_src_file_path = data_path + "/clean.id"
    new_src_file_path = data_path + "/multi.id"

    cur_tgt_file_path = data_path + "/clean.id"
    new_tgt_file_path = data_path + "/multi.id"

    model_path = args.model_path

    aug = naf.Sometimes([
        nac.RandomCharAug(action="insert", aug_word_p=0.3, aug_char_max=1),
        nac.RandomCharAug(action="delete", aug_word_p=0.3, aug_char_max=1),
        nac.RandomCharAug(action="substitute", aug_word_p=0.3, aug_char_max=1),
        nac.RandomCharAug(action="swap", aug_word_p=0.3, aug_char_max=1),
        naw.WordEmbsAug(model_type='fasttext', model_path=model_path, action="substitute", aug_p=0.3, top_k=1),
        naw.WordEmbsAug(model_type='fasttext', model_path=model_path, action="insert", aug_p=0.3, top_k=1),
        naw.RandomWordAug(action="swap", aug_p=0.3),
        naw.RandomWordAug(action="delete", aug_p=0.3)
    ], aug_p=0.125)

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

    # print("Copy the tgt files")

