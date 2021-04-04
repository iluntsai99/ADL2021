from typing import List, Dict

from torch.utils.data import Dataset
import torch

from utils import Vocab
from utils import pad_to_len


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    # def collate_fn(self, samples: List[Dict]) -> Dict:
    #     # TODO: implement collate_fn
    #     raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class slotDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # raise NotImplementedError
        tokens, text_lens, tags_list, ids = [], [], [], []
        for sample in samples:
            tokens.append(sample["tokens"])
            text_lens.append(len(sample["tokens"]))
            if "tags" in sample:
                tags_list.append(sample["tags"])
            ids.append(sample["id"])
        # X_batch = torch.LongTensor(self.vocab.encode_batch(tokens, self.max_len))
        X_batch = torch.LongTensor(self.vocab.encode_batch(tokens))
        text_lens = torch.LongTensor(text_lens)
            
        Y_batch = [[self.label_mapping[tag] for tag in tags] for tags in tags_list]
        # Y_batch = torch.LongTensor(pad_to_len(Y_batch, X_batch.shape[1], self.vocab.pad_id))
        Y_batch = torch.LongTensor(pad_to_len(Y_batch, X_batch.shape[1], self.vocab.pad_id))
        # print(X_batch.shape, Y_batch.shape)
        return X_batch, Y_batch, text_lens, ids

    def idx2label(self, idx: int):
        return self._idx2label[idx]
