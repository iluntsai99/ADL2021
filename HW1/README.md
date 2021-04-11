# README for Homework 1 ADL NTU 109 Spring

## Environment

```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# download models and data
bash download.sh
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
# training
python train_intent.py --data_dir <data_dir>
# inference
bash ./slot_tag.sh <test_file> <pred_file>
```

## Slot classification
```shell
# training
python train_slot.py --data_dir <data_dir>
# inference
bash ./slot_tag.sh <test_file> <pred_file>
```

**Note: if you want to add CRF layer, you need to comment the `nn.CrossEntropy` method and uncomment the CRF implementation.**

## SeqVal

```shell
python seqVal.py --test_file <eval_file> --ckpt_path <checkpoint_file>
```

## Testing steps for myself

`bash download.sh`

`bash ./intent_cls.sh ./data/intent/test.json i.csv`

`bash ./slot_tag.sh ./data/slot/test.json s.csv`

`vimdiff i.csv ../../../ADL2021/HW1/newInt.csv`

`vimdiff s.csv ../../../ADL2021/HW1/newSlot.csv `