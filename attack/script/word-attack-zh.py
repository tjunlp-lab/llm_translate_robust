from fasttext import load_model
import numpy as np
import jieba
import math
import random
import re
from tqdm import tqdm
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cur_data_path", type=str)
parser.add_argument("--new_data_path", type=str)
parser.add_argument("--model_path", type=str)

args = parser.parse_args()

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
model.matrix = model.matrix.to("cuda:1")
# fasttext_vec = fasttext_model.get_input_matrix()

# TODO: stopwords punctuation
def add_noise(word_list, action_list, noise_prob=0.1,
                min_modi_words=1, max_modi_words=5):

    word_list_length = len(word_list)
    modi_word_count = math.ceil(noise_prob * word_list_length)

    if modi_word_count > max_modi_words:
        modi_word_count = max_modi_words
        # raise Exception("Exceed the maxium number of modified characters")
    elif modi_word_count < min_modi_words:
        modi_word_count = min_modi_words
        # raise Exception("Not reach the minimum number of modified characters")

    selected_indices = random.sample(range(word_list_length), modi_word_count)

    for i in range(modi_word_count):
        cur_word = word_list[selected_indices[i]]

        cur_action = random.choice(action_list)

        if cur_action == "insert":
            word_list[selected_indices[i]] = insert(cur_word, top_k=10)
        elif cur_action == "delete":
            word_list[selected_indices[i]] = delete(cur_word)
        elif cur_action == "swap":
            target_word_indice = swap(selected_indices[i] - 1 if selected_indices[i] > 0 else None,
                                        selected_indices[i] + 1 if selected_indices[i] < word_list_length - 1 else None)
            
            if target_word_indice == None:
                continue

            tmp = word_list[selected_indices[i]]
            word_list[selected_indices[i]] = word_list[target_word_indice]
            word_list[target_word_indice] = tmp
        elif cur_action == "replace":
            word_list[selected_indices[i]] = replace(cur_word, top_k=10)

    return word_list

def is_chinese(word):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    result = pattern.findall(word)
    return len(result) == len(word)

def replace(word, top_k=1):
    if is_chinese(word) == False:
        return word

    candidate_word_list = model.get_nearest_words(model[word].to("cuda:1"), top_k)
    # candidate_word_list = fasttext_model.get_nearest_neighbors(word, k=1)
    replace_word = random.choice(candidate_word_list)
    return replace_word

def swap(word_indice1, word_indice2):
    if word_indice1 == None:
        target_word_indice = word_indice2
    elif word_indice2 == None:
        target_word_indice = word_indice1
    else:
        target_word_indice = random.choice([word_indice1, word_indice2])

    return target_word_indice

def delete(word):
    return ""

def insert(word, top_k=1):
    if is_chinese(word) == False:
        return word

    candidate_word_list = model.get_nearest_words(model[word].to("cuda:1"), top_k)
    # candidate_word_list = fasttext_model.get_nearest_neighbors(word, k=1)
    insert_word = random.choice(candidate_word_list)
    return random.choice([word + insert_word, insert_word + word])

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
        noisy_word_list = add_noise(word_list, action_list=["insert", "swap", "delete", "replace"],
                                    noise_prob=0.4, min_modi_words=1, max_modi_words=50)
        noisy_sent_list.append("".join(noisy_word_list))

    with open(new_data_path, 'w', encoding='utf-8') as file:
        for line in noisy_sent_list:
            if line.endswith('\n'):
                file.write(line)
            else:
                file.write(line + '\n')

    # for i in tqdm(range(20000)):
    #     input_text = "我喜欢吃中国菜，今天真是天气好"

    #     word_list = list(jieba.cut(input_text))

    #     print(word_list)

    #     noisy_word_list = add_noise(word_list, action_list=["insert"])

    #     noisy_text = "".join(noisy_word_list)

    #     print("原始文本:", input_text)
    #     print("噪音文本:", noisy_text)


