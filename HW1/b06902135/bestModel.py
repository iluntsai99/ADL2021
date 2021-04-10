from typing import Dict

import torch
from torch import nn
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=True)
        
        # TODO: model architecture
        self.gru = nn.GRU(embeddings.size(1), hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(hidden_size * 4, num_class),
                                         nn.Sigmoid())
        

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        inputs = self.embed(batch)
        x, _ = self.gru(inputs, None)  # x 的 dimension (batch, seq_len, hidden_size)
        a = x[:, -1, :] # get the end of hidden state
        b = x[:, 0, :] # get the start of hidden state
        c = torch.cat((a, b), dim=1)
        # print(c.shape)
        return self.classifier(c)

class slotClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        pad_id
    ) -> None:
        super(slotClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=True)
        
        # TODO: model architecture
        self.gru = nn.GRU(embeddings.size(1), hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(hidden_size * 2, hidden_size * 2),
                                         nn.Sigmoid(),
                                         nn.Dropout(dropout),
                                         nn.Linear(hidden_size * 2, num_class),)

        self.pack = lambda inputs, input_lens: pack_padded_sequence(inputs, input_lens, \
                                                            batch_first=True, enforce_sorted=False)
        self.unpack = lambda inputs: pad_packed_sequence(inputs, batch_first=True, padding_value=pad_id)
        

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        inputs = self.embed(batch)
        # print(batch.shape)
        # print(inputs.shape)
        x, _ = self.gru(inputs, None)  # x 的 dimension (batch, seq_len, hidden_size)
        # print(x.shape)
        # a = x[:, -1, :] # get the end of hidden state
        # b = x[:, 0, :] # get the start of hidden state
        # c = torch.cat((a, b), dim=1)
        # print(c.shape)
        return self.classifier(x)
    
    def compute(self, batch, text_lens):
        inputs = self.embed(batch)
        # print("compute", inputs.shape)
        packed_x, _ = self.gru(self.pack(inputs, text_lens.cpu()), None)  # x 的 dimension (batch, seq_len, hidden_size)
        x, _ = self.unpack(packed_x)
        # print("x ", x.shape)
        return self.classifier(x)
    
    def evaluation(self, outputs, text_lens, targets):
        # print(outputs)
        # print(targets)
        compare = torch.eq(outputs, targets)
        # print(compare[0][0:text_lens[0]])
        wrong = 0
        for i, len in enumerate(text_lens):
            # print(compare[i, 0:len])
            wrong += 1 if False in compare[i, 0:len] else 0
        correct = (text_lens.shape[0] - wrong)
        # print(correct)
        return correct