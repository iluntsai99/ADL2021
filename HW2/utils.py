import torch
import numpy as np
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def chunk(seq):
    offset = 512
    cur, cnt = 0, 0
    result = list()
    while cur < len(seq):
        result.append(seq[cur: cur + offset])
        cur += 450
        cnt += 1
    # print(result)
    return result, cnt

def evaluate(data, output, tokenized_paragraph, paragraph):
    answer = ''     
    max_answer_len = 35
    doc_stride = 350
    max_prob = float('-inf')
    num_windows = data[0].shape[1]
    paragraph_offset = data[1][0][0].tolist().index(1) # query + paragraph + pad, we want to clip query
    for k in range(num_windows):
        start_probs, start_indexs = torch.topk(output.start_logits[k], k=1, dim=0)
        for start_prob, start_index in zip(start_probs, start_indexs):
            length_prob, length = torch.max(output.end_logits[k][start_index : start_index + max_answer_len], dim=0)
            prob = start_prob + length_prob
            if prob > max_prob:
                max_prob = prob
                start_token_index = start_index - paragraph_offset + k * doc_stride
                try:
                    end_token_index = start_index + length - paragraph_offset + k * doc_stride
                    start_char_index = tokenized_paragraph.token_to_chars(start_token_index)[0]
                    end_char_index = tokenized_paragraph.token_to_chars(end_token_index)[1]
                    answer = paragraph[start_char_index : end_char_index]
                except:
                    pass
    if '「' in answer and '」' not in answer:
        answer += '」'
    elif '「' not in answer and '」' in answer:
        answer = '「' + answer
    if '《' in answer and '》' not in answer:
        answer += '》'
    elif '《' not in answer and '》' in answer:
        answer = '《' + answer
    return answer.replace(',','')