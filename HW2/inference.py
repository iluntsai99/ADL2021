import json
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, BertForMultipleChoice
from transformers import get_linear_schedule_with_warmup

import utils
from QA_Dataset import QA_Dataset
from CS_Dataset import CS_Dataset
from accelerate import Accelerator
import time
from utils import evaluate

TEST = "public"

def main(args):
    SPLITS = [TEST]
    context_path = args.context_path
    contexts = json.loads(context_path.read_text())
    data_paths = {split: args.data_path for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    
    CS_Model = BertForMultipleChoice.from_pretrained(args.ckpt_CS_dir).to(device)
    QA_model = BertForQuestionAnswering.from_pretrained(args.ckpt_QA_dir).to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
    
    context_tokenized = tokenizer(contexts, add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question"] for test_question in data[TEST]], add_special_tokens=False)
    test_CS_set = CS_Dataset(TEST, data[TEST], test_questions_tokenized, context_tokenized)
    test_CS_loader = DataLoader(test_CS_set, batch_size=1, shuffle=False)

    CS_Model.eval()
    with torch.no_grad():
        for i, datas in enumerate(tqdm(test_CS_loader)):
            output = CS_Model(input_ids=datas[0].to(device), token_type_ids=datas[1].to(device), attention_mask=datas[2].to(device))
            document = torch.argmax(output.logits, dim=1)
            if document >= len(data[TEST][i]["paragraphs"]):
                document = len(data[TEST][i]["paragraphs"]) - 1
            data[TEST][i]["relevant"] = data[TEST][i]["paragraphs"][document]
    
    QA_model.eval()
    test_QA_set = QA_Dataset(TEST, data[TEST], test_questions_tokenized, context_tokenized)
    test_QA_loader = DataLoader(test_QA_set, batch_size=1, shuffle=False)
    with torch.no_grad():
        result = dict()
        for i, datas in enumerate(tqdm(test_QA_loader)):
            output = QA_model(input_ids=datas[0].squeeze(dim=0).to(device), token_type_ids=datas[1].squeeze(dim=0).to(device),
                        attention_mask=datas[2].squeeze(dim=0).to(device))
            paragraph_id = data[TEST][i]["id"]
            relevant = data[TEST][i]["relevant"]
            result[paragraph_id] = evaluate(datas, output, context_tokenized[relevant], contexts[relevant])

    result_file = args.pred_path
    with open(result_file, 'w', encoding='utf-8') as f:	
        json.dump(result, f, ensure_ascii=False)
    print(f"Completed! Result is in {result_file}")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Directory to the dataset.",
    )
    parser.add_argument(
        "--ckpt_CS_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/ckpt_CS",
    )
    parser.add_argument(
        "--ckpt_QA_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/ckpt_QA",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Public or private",
    )
    parser.add_argument(
        "--pred_path",
        type=Path,
        help="Prediction file",
    )


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    utils.same_seeds(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    if "private" in str(args.data_path):
        TEST = "private"
    main(args)
