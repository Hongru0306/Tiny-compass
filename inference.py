import os
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
from model.LLM import internlm2Chat

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='internlm2')
    return parser.parse_args(args)

if __name__ == '__main__':
    os.chdir('your_root_path')
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]

    # datasets = ["multi_news", "multifieldqa_zh", "trec"]
    datasets = ["multifieldqa_zh"]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    pred_model = internlm2Chat(model2path[model_name], model_name)
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")

    for dataset in datasets:
        data = load_dataset('json', data_files=f'./dataset/{dataset}.jsonl',split='train')
        if not os.path.exists(f"pred/{model_name}"):
            os.makedirs(f"pred/{model_name}")
        out_path = f"pred/{model_name}/{dataset}.jsonl"
        if os.path.isfile(out_path):
            os.remove(out_path)
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        pred_model.get_pred(data, max_length, max_gen, prompt_format, device, out_path)   

