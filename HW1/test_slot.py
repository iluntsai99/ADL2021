import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import slotDataset
from bestModel import slotClassifier
from utils import Vocab
import utils
from tqdm import tqdm


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    print("loading testing data ...")
    data = json.loads(args.test_file.read_text())
    dataset = slotDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = args.batch_size, shuffle = False, num_workers = 8, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = slotClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        vocab.pad_id
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model

    # TODO: predict dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(ckpt, False)
    prediction = []
    id = []
    with torch.no_grad():
        for X_batch, _, text_lens, ids in tqdm(test_loader):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=-1).int().tolist()
            preds = [pred[:text_len] for pred, text_len in zip(preds, text_lens.tolist())]
            prediction += preds
            id += ids
    
    prediction = [' '.join([dataset.idx2label(label_id) for label_id in pred]) for pred in prediction]
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        f.write('id,tags\n')
        for i, result in zip(id, prediction):
            # print(dataset.idx2label(result))
            f.write('{},{}\n'.format(i, result))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        # required=True
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        # required=True
        default="./ckpt/slot/best.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    utils.same_seeds(0)
    args = parse_args()
    main(args)
