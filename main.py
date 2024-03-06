import argparse
from tqdm import tqdm
import random
import torch
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from transformers.generation.utils import GenerationConfig

separator = "ROUND [*****]\n"

task_level_prompt_zh = "Please remove the noise in the following sentences and translate the sentences from French to English:\n"
src_demonstration_template_zh = "xxxxx\n"
zh_pre_template = "The corresponding English translation is:\n"
tgt_demonstration_template_zh = zh_pre_template + "ooooo\n"


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model_name", default="None", type=str)
    parser.add_argument("--model_path", default="None", type=str)
    parser.add_argument("--noise_type", default="None", type=str)
    parser.add_argument("--task_type", default="None", type=str)
    parser.add_argument("--noise_data_type_name", default="None", type=str)
    parser.add_argument("--src_file_path", default="None", type=str)
    parser.add_argument("--tgt_file_path", default="None", type=str)
    parser.add_argument("--clean_src_dev_path", default="None", type=str)
    parser.add_argument("--character_src_dev_path", default="None", type=str)
    parser.add_argument("--word_src_dev_path", default="None", type=str)
    parser.add_argument("--multi_src_dev_path", default="None", type=str)
    parser.add_argument("--top_5_for_itself", default="None", type=str)
    parser.add_argument("--top_5_for_clean", default="None", type=str)
    parser.add_argument("--top_5_for_all", default="None", type=str)
    parser.add_argument("--res_path", default="None", type=str)
    parser.add_argument("--redundancy_res_path", default="None", type=str)
    parser.add_argument("--prompt_path", default="None", type=str)

    args = parser.parse_args()
    return args


def set_random_sed(seed_value):
    random.seed(seed_value)

def get_X_shot_prompt(src_prompt_list, tgt_prompt_list, src_text, few_shots, prompt_language="Chinese"):
    assert len(src_prompt_list) == len(tgt_prompt_list)

    if prompt_language == "Chinese":
        # prompt = task_level_prompt
        prompt = ""

        for i in range(len(src_prompt_list)):
            prompt = prompt + separator.replace("*****", str(i + 1))
            # if i == 0:
            prompt = prompt + task_level_prompt_zh
            prompt = prompt + src_demonstration_template_zh.replace("xxxxx", src_prompt_list[
                i]) + tgt_demonstration_template_zh.replace("ooooo", tgt_prompt_list[i])
            # prompt = prompt + ' ### '

        if few_shots > 0:
            prompt = prompt + separator.replace("*****", str(i + 2))
        prompt = prompt + task_level_prompt_zh
        prompt = prompt + src_demonstration_template_zh.replace("xxxxx", src_text)
        prompt = prompt + zh_pre_template

    return prompt


def get_response(args, prompt, tokenizer, model, model_type, few_shots):
    if model_type == 'chat':
        if args.model_name == 'baichuan7b':
            # baichuan-7b-chat
            messages = []
            messages.append({"role": "user", "content": prompt})
            redundancy_response = model.chat(tokenizer, messages)
        elif args.model_name == 'qwen7b':
            # qwen
            redundancy_response, history = model.chat(tokenizer, prompt, history=None)
        else:
            # chatglm, internlm
            redundancy_response, history = model.chat(tokenizer, prompt, history=[])
    else:
        inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs, max_new_tokens=256)
        redundancy_response = tokenizer.decode(outputs[0])

    response = get_clean_response(redundancy_response, few_shots)
    return redundancy_response, response

def get_clean_response(text, few_shots):
    pos = 0
    cnt = 0
    while True:
        pos = text.find(zh_pre_template, pos)
        if pos != -1:
            cnt += 1
            if cnt == few_shots + 1:
                break
            pos += 1
        else:
            break
    if cnt > 0:
        text = text[(pos+len(zh_pre_template)):].lstrip()
    return text


def get_data(file_path):
    text_list = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            text_list.append(line.replace("\n", ""))

    return text_list


def save_data(save_path, data_list):
    with open(save_path, "w", encoding="utf-8") as file:
        for line in data_list:
            file.write(line + "\n")


def load_test_dev_data(args, test_or_dev):
    file_path = args.src_file_path
    text_list = get_data(file_path)

    if test_or_dev == 'test':
        return text_list
    else:
        file_path = args.tgt_file_path
        tgt_text_list = get_data(file_path)
        return text_list, tgt_text_list

def load_all_dev_data(args):

    file_path = args.tgt_dev_path
    tgt_dev_list = get_data(file_path)

    file_path = args.clean_src_dev_list
    clean_src_dev_list = get_data(file_path)

    file_path = args.character_src_dev_list
    character_src_dev_list = get_data(file_path)

    file_path = args.word_src_dev_list
    word_src_dev_list = get_data(file_path)

    multi_src_dev_list = []

    return clean_src_dev_list, character_src_dev_list, word_src_dev_list, multi_src_dev_list, tgt_dev_list


def load_top_5_idx(args):
    index_list_for_itself = []
    index_list_for_clean = []
    index_list_for_all = []

    with open(args.top_5_for_itself, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            index_list_for_itself.append(data['top_index'])

    with open(args.top_5_for_clean, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            index_list_for_clean.append(data['top_index'])

    with open(args.top_5_for_all, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            index_list_for_all.append(data['top_index'])

    return index_list_for_clean, index_list_for_itself, index_list_for_all

def func(dev_src_list, dev_tgt_list, index_list):
    src_prompt_list = []
    tgt_prompt_list = []
    for i in range(len(index_list)):
        cur_src_prompt_list = []
        cur_tgt_prompt_list = []
        for index in index_list[i]:
            cur_src_prompt_list.append(dev_src_list[index])
            cur_tgt_prompt_list.append(dev_tgt_list[index])
        src_prompt_list.append(cur_src_prompt_list)
        tgt_prompt_list.append(cur_tgt_prompt_list)
    return src_prompt_list, tgt_prompt_list


def return_dev_by_noise(clean_src_dev_list, character_src_dev_list, word_src_dev_list, multi_src_dev_list, noise_type):
    if noise_type == 'clean':
        return clean_src_dev_list
    elif noise_type == 'character':
        return character_src_dev_list
    elif noise_type == 'word':
        return word_src_dev_list
    elif noise_type == 'multi':
        return multi_src_dev_list


def return_list_by_id(res_list_0, res_list_1, res_list_3, res_list_5, id):
    if id == 0:
        return res_list_0
    elif id == 1:
        return res_list_1
    if id == 3:
        return res_list_3
    elif id == 5:
        return res_list_5


def evaluate_data(args, tokenizer, model, model_type, noise_data_type, spec_or_random, model_name):

    assert spec_or_random in ['SPECIFY', 'RANDOM', 'CLEAN']
    test_src_list = load_test_dev_data(args, test_or_dev='test')
    clean_src_dev_list, character_src_dev_list, word_src_dev_list, multi_src_dev_list, tgt_dev_list = load_all_dev_data(args)
    all_src_dev_list = clean_src_dev_list + character_src_dev_list + word_src_dev_list + multi_src_dev_list
    all_tgt_dev_list = tgt_dev_list + tgt_dev_list + tgt_dev_list + tgt_dev_list

    index_list_for_clean, index_list_for_itself, index_list_for_all = load_top_5_idx(args)

    print('clean src dev sample count:{}, character src dev sample count:{}, word src dev sample count:{}, multi src dev sample count:{}'.format(
        len(clean_src_dev_list), len(character_src_dev_list), len(word_src_dev_list), len(multi_src_dev_list)))
    print('all src dev sample count:{}, all tgt dev sample count:{}'.format(len(all_src_dev_list), len(all_tgt_dev_list)))
    print('test sample count:{}'.format(len(test_src_list)))
    print('top_index_list_for_clean:{}, top_index_list_for_itself:{}, top_index_list_for_all:{}'.format(
        len(index_list_for_clean), len(index_list_for_itself), len(index_list_for_all)
    ))

    res_list_0 = []   # 去掉prompt的输出
    redundancy_res_list_0 = []    # 模型原始的输出(会带有输入的prompt)
    prompt_list_0 = []

    res_list_1 = []
    redundancy_res_list_1 = []
    prompt_list_1 = []

    res_list_3 = []
    redundancy_res_list_3 = []
    prompt_list_3 = []

    res_list_5 = []
    redundancy_res_list_5 = []
    prompt_list_5 = []


    for i in tqdm(range(len(test_src_list)), desc=noise_data_type):
        src_5_prompt = []
        tgt_5_prompt = []
        if spec_or_random == 'CLEAN':
            for index in index_list_for_clean[i]:
                src_5_prompt.append(clean_src_dev_list[index])
                tgt_5_prompt.append(tgt_dev_list[index])
        elif spec_or_random == 'RANDOM':
            for index in index_list_for_all[i]:
                src_5_prompt.append(all_src_dev_list[index])
                tgt_5_prompt.append(all_tgt_dev_list[index])
        else:
            tmp_dev_list = return_dev_by_noise(clean_src_dev_list, character_src_dev_list, word_src_dev_list,
                                               multi_src_dev_list,
                                               noise_type=noise_data_type)
            for index in index_list_for_itself[i]:
                src_5_prompt.append(tmp_dev_list[index])
                tgt_5_prompt.append(tgt_dev_list[index])

        assert len(src_5_prompt) == 5
        assert len(tgt_5_prompt) == 5

        for shots in [0, 3]:
            if shots == 0:
                if spec_or_random != 'SPECIFY':
                    continue
            src_prompt_list = []
            tgt_prompt_list = []
            if shots > 0:
                src_prompt_list = src_5_prompt[:shots]
                tgt_prompt_list = tgt_5_prompt[:shots]

            prompt = get_X_shot_prompt(src_prompt_list, tgt_prompt_list, test_src_list[i], shots)
            redundancy_response, response = get_response(args, prompt, tokenizer, model, model_type=model_type,
                                                         few_shots=shots)
            if shots == 0:
                res_list_0.append(response.replace("\r", " ").replace("\n", " "))
                redundancy_res_list_0.append(redundancy_response.replace("\r", " ").replace("\n", " "))
                prompt_list_0.append(prompt)
            elif shots == 1:
                res_list_1.append(response.replace("\r", " ").replace("\n", " "))
                redundancy_res_list_1.append(redundancy_response.replace("\r", " ").replace("\n", " "))
                prompt_list_1.append(prompt)
            elif shots == 3:
                res_list_3.append(response.replace("\r", " ").replace("\n", " "))
                redundancy_res_list_3.append(redundancy_response.replace("\r", " ").replace("\n", " "))
                prompt_list_3.append(prompt)
            elif shots == 5:
                res_list_5.append(response.replace("\r", " ").replace("\n", " "))
                redundancy_res_list_5.append(redundancy_response.replace("\r", " ").replace("\n", " "))
                prompt_list_5.append(prompt)

    for shots in [0, 3]:
        if shots == 0:
            if spec_or_random != 'SPECIFY':
                continue
        res_path = args.res_path + "/{}.{}.{}.{}.res".format(
            model_name, args.noise_type, args.task_type, shots, spec_or_random, noise_data_type
        )
        redundancy_res_path = args.redundancy_res_path + "/{}.{}.{}.{}.redundancyres".format(
            model_name, args.noise_type, args.task_type, shots, spec_or_random, noise_data_type
        )
        prompt_path = args.prompt_path + "/{}.{}.{}.{}.prompt".format(
            model_name, args.noise_type, args.task_type, shots, spec_or_random, noise_data_type
        )

        res_list = return_list_by_id(res_list_0, res_list_1, res_list_3, res_list_5, shots)
        save_data(res_path, res_list)

        redundancy_res_list = return_list_by_id(redundancy_res_list_0, redundancy_res_list_1, redundancy_res_list_3, redundancy_res_list_5, shots)
        save_data(redundancy_res_path, redundancy_res_list)

        prompt_list = return_list_by_id(prompt_list_0, prompt_list_1, prompt_list_3, prompt_list_5, shots)
        save_data(prompt_path, prompt_list)


if __name__ == "__main__":

    _args = args()
    set_random_sed(_args.seed)
    model_path = _args.model_path
    model_name = _args.model_name
    noise_data_type_name = _args.noise_data_type_name

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)


    if _args.noise_type == 'sythetic':
        # # CLEAN
        print('------------------CLEAN------------------')
        evaluate_data(_args, tokenizer, model, model_type='base', noise_data_type=noise_data_type_name, spec_or_random='CLEAN', model_name=model_name)
        # # RANDOM
        print('------------------RANDOM------------------')
        evaluate_data(_args, tokenizer, model, model_type='base', noise_data_type=noise_data_type_name, spec_or_random='RANDOM', model_name=model_name)
        # SPECIFY
        print('------------------SPECIFY------------------')
        evaluate_data(_args, tokenizer, model, model_type='base', noise_data_type=noise_data_type_name, spec_or_random='SPECIFY', model_name=model_name)

