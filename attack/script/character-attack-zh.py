import jieba
import random
import math
from pypinyin import pinyin
from pypinyin.contrib.tone_convert import to_normal
import re
from tqdm import tqdm
from fasttext import load_model
import numpy as np
import argparse

homophone_file_path = "Chinese-Noisy-Text/ChineseHomophones/chinese_homophone_char.txt"
homophone_char_dict = {}
with open(homophone_file_path, "r", encoding="utf-8") as file:
    for line in file.readlines():
        line_list = line.replace("\n", "").split("\t")
        homophone_char_dict[line_list[0]] = line_list[1:]

# TODO: stopwords punctuation 
# 定义字符级别的噪音增强函数
def add_noise(word_list, action_list, noise_prob=0.1, 
                min_modi_words=1, max_modi_words=5,
                char_oper_count=1):
    word_list_length = len(word_list)
    modi_word_count = math.ceil(noise_prob * word_list_length)
    # print(modi_char_count)

    if modi_word_count > max_modi_words:
        modi_word_count = max_modi_words
        # raise Exception("Exceed the maxium number of modified characters")
    elif modi_word_count < min_modi_words:
        modi_word_count = min_modi_words
        # raise Exception("Not reach the minimum number of modified characters")

    selected_indices = random.sample(range(word_list_length), modi_word_count)
    # print(selected_words)
    
    for i in range(modi_word_count):
        cur_word = word_list[selected_indices[i]]
        cur_action = random.choice(action_list)
        for j in range(char_oper_count):
            if cur_action == "character_insert":
                word_list[selected_indices[i]] = character_insert(cur_word)
            elif cur_action == "character_delete":
                word_list[selected_indices[i]] = character_delete(cur_word)
            elif cur_action == "character_swap":
                word_list[selected_indices[i]] = character_swap(cur_word)
            elif cur_action == "character_replace":
                word_list[selected_indices[i]] = character_replace(cur_word)
    
    return word_list


def character_delete(word):
    word_length = len(word)
    if word_length < 2:
        return word

    position = random.randint(0, word_length - 1)

    return word[:position] + word[position + 1:]


def character_swap(word):
    word_length = len(word)
    char_list = list(word)
    if word_length < 2:
        return word

    position = random.randint(0, word_length - 1)

    if position == 0:
        candidate_char_list = [0, 1]
    elif position == word_length - 1:
        candidate_char_list = [word_length - 2, word_length - 1]
    else:
        candidate_char_list = [position, random.choice([position - 1, position + 1])]

    tmp = char_list[candidate_char_list[0]]
    char_list[candidate_char_list[0]] = char_list[candidate_char_list[1]]
    char_list[candidate_char_list[1]] = tmp

    return "".join(char_list)

def character_replace(word):
    if is_chinese(word) == False:
        return word
    word_length = len(word)
    char_list = list(word)

    position = random.randint(0, word_length - 1)
    char_list[position] = getHomophone(char_list[position])

    return "".join(char_list)


def character_insert(word):
    if is_chinese(word) == False:
        return word

    word_length = len(word)

    position = random.randint(0, word_length)
    # print(position)

    if position == 0:
        target_char_list = [word[0]]
    elif position == word_length:
        target_char_list = [word[word_length - 1]]
    else:
        target_char_list = [word[position - 1], word[position]]

    target_char = random.choice(target_char_list)
    insert_char = getHomophone(target_char)
    
    return word[:position] + insert_char + word[position:]
    

def getHomophone(target_char):
    target_char_pinyin = to_normal(pinyin(target_char)[0][0])
    if target_char_pinyin not in homophone_char_dict:
        return target_char
    candidate_char_list = homophone_char_dict[target_char_pinyin]
    choose_char = random.choice(candidate_char_list)
    return choose_char

def is_chinese(word):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    result = pattern.findall(word)
    return len(result) == len(word)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cur_data_path", type=str)
    parser.add_argument("--new_data_path", type=str)

    args = parser.parse_args()

    cur_data_path = args.cur_data_path
    new_data_path = args.new_data_path

    sent_list = []
    with open(cur_data_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            sent_list.append(line.replace("\n", ""))

    noisy_sent_list = []
    for i in tqdm(range(len(sent_list))):
        word_list = list(jieba.cut(sent_list[i]))
        noisy_word_list = add_noise(word_list, action_list=["character_insert", "character_swap", "character_delete", "character_replace"],
                                    noise_prob=0.5, min_modi_words=1, max_modi_words=50)
        noisy_sent_list.append("".join(noisy_word_list))

    with open(new_data_path, 'w', encoding='utf-8') as file:
        for line in noisy_sent_list:
            if line.endswith('\n'):
                file.write(line)
            else:
                file.write(line + '\n')


    # input_text = "我喜欢吃中国菜，今天真是天气好"

    # word_list = list(jieba.cut(input_text))

    # print(word_list)

    # noisy_word_list = add_noise(word_list, action_list=["character_delete"])

    # noisy_text = "".join(noisy_word_list)

    # print("原始文本:", input_text)
    # print("噪音文本:", noisy_text)