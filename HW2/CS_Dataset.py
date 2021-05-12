from torch.utils.data import DataLoader, Dataset, ConcatDataset
import random
import torch

class CS_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.num_choices = 7
        self.max_question_len = 40
        self.max_paragraph_len = 469

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        question_ID = question["id"]
        related_IDs = question["paragraphs"]
        tokenized_question = self.tokenized_questions[index]
        tokenized_paragraph = [""]*7
        # print("related len", len(related_IDs))
        # print("tokenized_paragraph len", len(tokenized_paragraph))
        for i, paragraph in enumerate(related_IDs):
            tokenized_paragraph[i] = self.tokenized_paragraphs[paragraph]
        # print("question_ID", question_ID)
        # print("related ID", related_IDs)

        if self.split == "train" or self.split == "public":
            relevant = question["relevant"]
            # print(related_IDs)
            label = related_IDs.index(relevant)
            # print(question['question'], relevant, label)
            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = list()
            for i in range(self.num_choices):
                input_ids_paragraph += [tokenized_paragraph[i].ids[:self.max_paragraph_len] + [102]] if i < len(related_IDs) else [[]]
            # print(len(input_ids_paragraph))
            # Pad sequence and obtain inputs to model
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            # print(input_ids.shape, token_type_ids.shape, attention_mask.shape)
            return input_ids, token_type_ids, attention_mask, label

        # Validation/Testing
        else:
            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = list()
            for i in range(self.num_choices):
                input_ids_paragraph += [tokenized_paragraph[i].ids[:self.max_paragraph_len] + [102]] if i < len(related_IDs) else [[]]
            
            # Pad sequence and obtain inputs to model
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            # print(input_ids.shape, token_type_ids.shape, attention_mask.shape)
            return input_ids, token_type_ids, attention_mask

    def padding(self, input_ids_question, input_ids_paragraph):
        input_ids = list()
        token_type_ids = list()
        attention_mask = list()
        for i, paragraph in enumerate(input_ids_paragraph):
            # Pad zeros if sequence length is shorter than max_seq_len
            padding_len = self.max_seq_len - len(input_ids_question) - len(paragraph)
            # print(self.max_seq_len, len(input_ids_question), len(paragraph), padding_len)
            # Indices of input sequence tokens in the vocabulary
            input_ids.append(input_ids_question + paragraph + [0] * padding_len)
            # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
            token_type_ids.append([0] * len(input_ids_question) + [1] * len(paragraph) + [0] * padding_len)
            # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
            attention_mask.append([1] * (len(input_ids_question) + len(paragraph)) + [0] * padding_len)
        # print(input_ids, token_type_ids, attention_mask)
        # return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)