import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm
from tqdm import trange

from dataset import SeqClsDataset
from bestModel import SeqClassifier
from utils import Vocab
import utils
import numpy as np
import time
import copy

TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]
batch_size = 64
lr = 1e-3
hidden_size = 512
num_layers = 2
dropout = 0.5
bidirectional = True
do_semi = False

def get_pseudo_labels(dataset, model, vocab, device, threshold=0.65):
    test_loader = torch.utils.data.DataLoader(dataset=dataset[TEST], batch_size=batch_size, shuffle=False, num_workers=8)
    model.eval()
    softmax = nn.Softmax(dim=-1)

    count = 0
    indice = []
    Categ = []
    indice = np.array(indice)
    Categ = np.array(Categ)
    with torch.no_grad():
        for idx, inputs in enumerate(test_loader):
            # print(inputs)
            X_batch = [sentence.split(" ") for sentence in inputs['text']]
            X_batch = torch.LongTensor(vocab.encode_batch(X_batch)).to(device)
            outputs = model(X_batch)
            probs = softmax(outputs)
            max, category = torch.max(probs, -1)
            max = max.cpu().numpy()
            print(max[0])
            category = category.cpu().numpy()

            addIdx = np.argwhere(max > threshold)
            indice = np.append(indice, addIdx + count * batch_size)
            # print( addIdx + count * batch_size)
            # for i in addIdx + count * batch_size:
            #     print(i)
            for i in addIdx:
                Categ = np.append(Categ, int(category[i]))
            count += 1

    print("adding {0} of testing datas with thresh {1}".format(indice.shape[0], threshold))
    model.train()
    addDataset = copy.deepcopy(Subset(dataset[TEST], indice.astype(int)))
    for i in range(len(addDataset)):
        addDataset[i]['intent'] = dataset[TEST].idx2label(Categ[i])
    # for idx, data in enumerate(addDataset):
    #     print(idx, data)
    # print(addDataset[0]['intent'])
    # print(addDataset[-1]['intent'])
    return addDataset

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    addTrain_size = int(0.8 * len(datasets[DEV]))
    valid_size = len(datasets[DEV]) - addTrain_size
    addTrain_dataset, datasets[DEV] = torch.utils.data.random_split(datasets[DEV], [addTrain_size, valid_size])
    datasets[TRAIN] = ConcatDataset([datasets[TRAIN], addTrain_dataset])

    train_loader = torch.utils.data.DataLoader(dataset=datasets[TRAIN], batch_size=batch_size, shuffle=True, num_workers=8)
    dev_loader = torch.utils.data.DataLoader(dataset=datasets[DEV], batch_size=batch_size, shuffle=False, num_workers=8)
    
    # print(datasets[TRAIN][0])
    # print(datasets[TRAIN][0]['text'])
    # print(vocab.encode(datasets[TRAIN][0]['text'].split(" ")))
    # print(len(intent2idx))

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqClassifier(embeddings, hidden_size, num_layers, dropout, bidirectional, len(intent2idx)).to(device)
    print(model)
    # TODO: init optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_size = 0, 0
    # threshold = [0.99, 0.95, 0.9]
    # start_semi = 10
    early_stop = 20
    lr_now = lr
    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in range(args.num_epoch):
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        # do_semi = True if epoch >= start_semi else False

        if (do_semi):
            pseudo_set = get_pseudo_labels(datasets, model, vocab, device, threshold=threshold[int(epoch / (args.num_epoch / len(threshold)))])
            concat_dataset = ConcatDataset([datasets[TRAIN], pseudo_set])
            train_loader = torch.utils.data.DataLoader(dataset=concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        
        train_size, dev_size = len(train_loader.dataset), len(dev_loader.dataset)
        print(train_size, dev_size)
        model.train()
        total_loss, total_acc = 0, 0
        start_time = time.time()
        for inputs in tqdm(train_loader):
            X_batch = [sentence.split(" ") for sentence in inputs['text']]
            Y_batch = [intent2idx[label] for label in inputs['intent']]
            # print(X_batch)
            # print(Y_batch)
            X_batch = torch.LongTensor(vocab.encode_batch(X_batch)).to(device)
            Y_batch = torch.tensor(Y_batch).to(device)
            # print("hoho", X_batch)
            # print("hoho", Y_batch)
            optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward() # 算 loss 的 gradient
            optimizer.step()
            # print(Y_batch)
            correct = utils.evaluation(outputs.argmax(dim=-1), Y_batch) 
            total_acc += correct
            total_loss += loss.item()
        print('[ Epoch {}/{}] {:2.2f}s Train | Loss:{:.5f} Acc: {:.3f}'.format(epoch, args.num_epoch, time.time() - start_time, total_loss / train_size, total_acc / train_size))
        
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            start_time = time.time()
            for inputs in tqdm(dev_loader):
                X_batch = [sentence.split(" ") for sentence in inputs['text']]
                Y_batch = [intent2idx[label] for label in inputs['intent']]
                X_batch = torch.LongTensor(vocab.encode_batch(X_batch)).to(device)
                Y_batch = torch.tensor(Y_batch).to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                correct = utils.evaluation(outputs.argmax(dim=-1), Y_batch) 
                total_acc += correct
                total_loss += loss.item()
        print('[ Epoch {}/{}] {:2.2f}s Valid | Loss:{:.5f} Acc: {:.3f}'.format(epoch, args.num_epoch, time.time() - start_time, total_loss / dev_size, total_acc / dev_size))
        if (train_size > best_size or (total_acc / dev_size) > best_acc):
        # if ((total_acc / dev_size) > best_acc):
            print("saving model...with {} data and acc: {}".format(train_size, (total_acc / dev_size)))
            best_acc = total_acc / dev_size
            best_size = train_size
            torch.save(model.state_dict(), args.ckpt_dir / f"best.pt")
            early_stop = 0
        else:
            early_stop += 1
        schedular.step(total_loss)
        if optimizer.param_groups[0]["lr"] != lr_now:
            print("Learning rate is changed to {} from {}".format(optimizer.param_groups[0]["lr"], lr_now))
            lr_now = optimizer.param_groups[0]["lr"]
        if lr_now < 1e-6 or early_stop > 20:
            print("Early stop!")
            break

        model.train()
    print("Saving with best acc: {}".format(best_acc))
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=lr)

    # data loader
    parser.add_argument("--batch_size", type=int, default=batch_size)

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


