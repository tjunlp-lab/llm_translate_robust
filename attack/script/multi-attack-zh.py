import jieba
import random
import math
from pypinyin import pinyin
from pypinyin.contrib.tone_convert import to_normal
import re
from tqdm import tqdm
from fasttext import load_model
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cur_data_path", type=str)
parser.add_argument("--new_data_path", type=str)
parser.add_argument("--model_path", type=str)

args = parser.parse_args()

homophone_file_path = "Chinese-Noisy-Text/ChineseHomophones/chinese_homophone_char.txt"
homophone_char_dict = {}
with open(homophone_file_path, "r", encoding="utf-8") as file:
    for line in file.readlines():
        line_list = line.replace("\n", "").split("\t")
        homophone_char_dict[line_list[0]] = line_list[1:]

class FastVector:
    def __init__(self, vector_file) -> None:
        self.model = load_model(vector_file)
        self.matrix = torch.zeros((len(self.model.words), self.model.get_dimension()))
        for i, word in tqdm(enumerate(self.model.words)):
            word_vector = self.model.get_word_vector(word)
            word_vector = torch.tensor(word_vector)
            word_vector = word_vector * (1.0 / torch.norm(word_vector))
            self.matrix[i] = word_vector

    def get_nearest_words(self, zh_word_vector, topn):
        zh_word_vector = zh_word_vector*(1.0 / torch.norm(zh_word_vector))
        sim_vec = torch.matmul(zh_word_vector, torch.transpose(self.matrix, 0, 1))
        _, top_indices = torch.topk(sim_vec, topn+1)
        candidate_word_list = [self.model.words[top_indices[i]] for i in range(len(top_indices))]
        # sorted_sim_vec = sorted(zip(self.model.words, sim_vec), key=lambda x:x[1], reverse=True)
        return candidate_word_list[1:topn+1]
    
    def __getitem__(self, word):
        return torch.tensor(self.model.get_word_vector(word))

    def __contains__(self, word):
        return word in self.model.words

model_path = args.model_path
model = FastVector(model_path)
model.matrix = model.matrix.to("cuda:2")
# fasttext_vec = fasttext_model.get_input_matrix()

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

        # character_action_list = []
        # word_action_list = []
        # for i in range(len(action_list)):
        #     if action_list[i].split("_")[0] == "character":
        #         character_action_list.append(action_list[i])
        #     elif action_list[i].split("_")[0] == "word":
        #         word_action_list.append(action_list[i])

        cur_action = random.choice(action_list)
        # print(cur_action)
        if cur_action.split("_")[0] == "character":
            for j in range(char_oper_count):
                if cur_action == "character_insert":
                    word_list[selected_indices[i]] = character_insert(cur_word)
                elif cur_action == "character_delete":
                    word_list[selected_indices[i]] = character_delete(cur_word)
                elif cur_action == "character_swap":
                    word_list[selected_indices[i]] = character_swap(cur_word)
                elif cur_action == "character_replace":
                    word_list[selected_indices[i]] = character_replace(cur_word)
        elif cur_action.split("_")[0] == "word":
            if cur_action == "word_insert":
                word_list[selected_indices[i]] = word_insert(cur_word, top_k=10)
            elif cur_action == "word_delete":
                word_list[selected_indices[i]] = word_delete(cur_word)
            elif cur_action == "word_swap":
                target_word_indice = word_swap(selected_indices[i] - 1 if selected_indices[i] > 0 else None,
                                            selected_indices[i] + 1 if selected_indices[i] < word_list_length - 1 else None)
                
                if target_word_indice == None:
                    continue

                tmp = word_list[selected_indices[i]]
                word_list[selected_indices[i]] = word_list[target_word_indice]
                word_list[target_word_indice] = tmp
            elif cur_action == "word_replace":
                word_list[selected_indices[i]] = word_replace(cur_word, top_k=10)
    
    return word_list

def word_replace(word, top_k=10):
    if is_chinese(word) == False:
        return word

    # word_vec = fasttext_vec[fasttext_model.get_word_id(word)]
    # sim_vec = np.dot(word_vec, fasttext_vec.T)
    # sorted_sim_vec = sorted(zip(fasttext_model.get_words(), sim_vec), key=lambda x:x[1], reverse=True)
    
    # candidate_word_list = sorted_sim_vec[1:top_k+1]
    # # print(candidate_word_list)
    # replace_word = random.choice(candidate_word_list)[0]
    # return replace_word
    
    candidate_word_list = model.get_nearest_words(model[word].to("cuda:2"), top_k)
    # candidate_word_list = fasttext_model.get_nearest_neighbors(word, k=1)
    replace_word = random.choice(candidate_word_list)
    return replace_word

def word_swap(word_indice1, word_indice2):
    if word_indice1 == None:
        target_word_indice = word_indice2
    elif word_indice2 == None:
        target_word_indice = word_indice1
    else:
        target_word_indice = random.choice([word_indice1, word_indice2])

    return target_word_indice

def word_delete(word):
    return ""

def word_insert(word, top_k=10):
    if is_chinese(word) == False:
        return word

    candidate_word_list = model.get_nearest_words(model[word].to("cuda:2"), top_k)
    # candidate_word_list = fasttext_model.get_nearest_neighbors(word, k=1)
    insert_word = random.choice(candidate_word_list)
    return random.choice([word + insert_word, insert_word + word])


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
    cur_data_path = args.cur_data_path
    new_data_path = args.new_data_path

    sent_list = []
    with open(cur_data_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            sent_list.append(line.replace("\n", ""))

    noisy_sent_list = []
    for i in tqdm(range(len(sent_list))):
        word_list = list(jieba.cut(sent_list[i]))
        noisy_word_list = add_noise(word_list, action_list=["character_insert", "character_swap", "character_delete", "character_replace", "word_insert", "word_swap", "word_delete", "word_replace"],
                                    noise_prob=0.4, min_modi_words=1, max_modi_words=50)
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

    # noisy_word_list = add_noise(word_list, action_list=["character_insert", "character_swap", "character_delete", "character_replace", "word_insert", "word_swap", "word_delete", "word_replace"])

    # noisy_text = "".join(noisy_word_list)

    # print("原始文本:", input_text)
    # print("噪音文本:", noisy_text)