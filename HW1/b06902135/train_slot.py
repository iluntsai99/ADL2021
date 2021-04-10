import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torchCRF import CRF
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm
from tqdm import trange

from dataset import slotDataset
from bestModel import slotClassifier
from utils import Vocab
import utils
import numpy as np
import time
import copy

TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]
batch_size = 16
lr = 1e-3
hidden_size = 512
num_layers = 2
dropout = 0.5
bidirectional = True

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, slotDataset] = {
        split: slotDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    collate_fn=datasets[TRAIN].collate_fn

    addTrain_size = int(0.8 * len(datasets[DEV]))
    valid_size = len(datasets[DEV]) - addTrain_size
    addTrain_dataset, datasets[DEV] = torch.utils.data.random_split(datasets[DEV], [addTrain_size, valid_size])
    datasets[TRAIN] = ConcatDataset([datasets[TRAIN], addTrain_dataset])
    
    # TODO: crecate DataLoader for train / dev datasets
    train_loader = torch.utils.data.DataLoader(dataset=datasets[TRAIN], batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
    dev_loader = torch.utils.data.DataLoader(dataset=datasets[DEV], batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
    # print(datasets[TRAIN][0:10])
    # print(datasets[TRAIN][0]["tokens"])
    # print(vocab.encode(datasets[TRAIN][0]["tokens"]))

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = slotClassifier(embeddings, hidden_size, num_layers, dropout, bidirectional, len(tag2idx), vocab.pad_id).to(device)

    # TODO: init optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    criterion = nn.CrossEntropyLoss()
    # criterion = CRF(len(tag2idx), batch_first=True).to(device)
    best_acc = 0

    early_stop = 20
    lr_now = lr
    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in range(args.num_epoch):
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        train_size, dev_size = len(train_loader.dataset), len(dev_loader.dataset)
        print(train_size, dev_size)
        model.train()
        total_loss, total_acc = 0, 0
        start_time = time.time()
        for X_batch, Y_batch, text_lens, _ in tqdm(train_loader):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            text_lens = text_lens.to(device)
            # print(X_batch, Y_batch, text_lens, ids)

            optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            outputs = model.compute(X_batch, text_lens)
            # print(outputs.shape)
            # print(Y_batch.shape)
            # print(outputs.reshape(-1, len(tag2idx)).shape)
            # print(Y_batch.reshape(-1).shape)
            loss = criterion(outputs.reshape(-1, len(tag2idx)), Y_batch.reshape(-1)) / text_lens.sum()
            # mask = torch.arange(X_batch.shape[1]).repeat(X_batch.shape[0], 1).to(device) < text_lens.unsqueeze(1)
            # loss = -criterion(outputs, Y_batch, mask=mask, reduction="mean") / text_lens.sum()
            loss.backward() # 算 loss 的 gradient
            optimizer.step()
            correct = model.evaluation(outputs.argmax(dim=-1), text_lens, Y_batch) 
            total_acc += correct
            total_loss += loss.item() * Y_batch.shape[0]
        print('[ Epoch {}/{}] {:2.2f}s Train | Loss:{:.5f} Acc: {:.3f}'.format(epoch, args.num_epoch, time.time() - start_time, total_loss / train_size, total_acc / train_size))
    
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            start_time = time.time()
            for X_batch, Y_batch, text_lens, _ in tqdm(dev_loader):
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                text_lens = text_lens.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs.reshape(-1, len(tag2idx)), Y_batch.reshape(-1)) / text_lens.sum()
                # mask = torch.arange(X_batch.shape[1]).repeat(X_batch.shape[0], 1).to(device) < text_lens.unsqueeze(1)
                # loss = -criterion(outputs, Y_batch, mask=mask, reduction="sum") / text_lens.sum()
                correct = model.evaluation(outputs.argmax(dim=-1), text_lens, Y_batch) 
                total_acc += correct
                total_loss += loss.item() * Y_batch.shape[0]
        print('[ Epoch {}/{}] {:2.2f}s Valid | Loss:{:.5f} Acc: {:.3f}'.format(epoch, args.num_epoch, time.time() - start_time, total_loss / dev_size, total_acc / dev_size))
        if (total_acc / dev_size) >= best_acc:
            print("saving model...with {} data and acc: {}".format(train_size, (total_acc / dev_size)))
            best_acc = total_acc / dev_size
            torch.save(model.state_dict(), args.ckpt_dir / f"plz.pt")
            early_stop = 0
        else:
            early_stop += 1
        schedular.step(total_loss)
        if optimizer.param_groups[0]["lr"] != lr_now:
            print("Learning rate is changed to {} from {}".format(optimizer.param_groups[0]["lr"], lr_now))
            lr_now = optimizer.param_groups[0]["lr"]
        # if lr_now < 1e-6 or early_stop > 20:
        if lr_now < 1e-6:
            print("Early stop!")
            break
    print("Saving with best acc: {}".format(best_acc))
    # TODO: Inference on test set

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=200)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    utils.same_seeds(0)
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)


