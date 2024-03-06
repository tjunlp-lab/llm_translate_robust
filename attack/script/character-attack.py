import nlpaug.augmenter.char as nac
import nlpaug.flow as naf
from tqdm import tqdm
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)

    args = parser.parse_args()

    aug = naf.Sometimes([
        nac.RandomCharAug(action="insert", aug_word_p=0.5, aug_char_max=1),
        nac.RandomCharAug(action="delete", aug_word_p=0.5, aug_char_max=1),
        nac.RandomCharAug(action="substitute", aug_word_p=0.5, aug_char_max=1),
        nac.RandomCharAug(action="swap", aug_word_p=0.5, aug_char_max=1)
    ], aug_p=0.25)

    data_path = args.data_path

    cur_src_file_path = data_path + "/clean.en"
    new_src_file_path = data_path + "/character.en"

    cur_tgt_file_path = data_path + "/clean.zh"
    new_tgt_file_path = data_path + "/character.zh"

    print("Attack the src files")

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
