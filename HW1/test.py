import torch
from torchCRF import CRF


num_tags = 5  # number of tags is 5
model = CRF(num_tags)
seq_length = 3  # maximum sequence length in a batch
batch_size = 2  # number of samples in the batch
emissions = torch.randn(seq_length, batch_size, num_tags)
tags = torch.tensor([[0, 1], [2, 4], [3, 1]], dtype=torch.long)  # (seq_length, batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
emissions = emissions.to(device)
tags = tags.to(device)
# print(emissions, tags)
print(model(emissions, tags, reduction="sum"))