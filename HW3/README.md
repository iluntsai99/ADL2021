# README for Homework 3 ADL NTU 109 Spring

## Environment

```shell
# all transformer package should be installed
pip install transformers
# if you want to use fp16 acceleration
pip install accelerate
```

## Preprocessing
```shell
# download models and configurations
bash download.sh
```

## Train
```shell
# training
python train.py --data_dir <data_dir>
# inference
bash run.sh <data_path> <prediction_path>
```

## Testing steps for myself

`bash download.sh`

`unzip data.zip`

`bash ./run.sh data/public.jsonl ./public_result.jsonl`

`python ./ADL21-HW3/eval.py -r ./public_result.josnl -s ../data/public.jsonl`

**Public score should be:**

```json
{
    "rouge-1": {
      "f": 0.24075470957606235,
      "p": 0.25188147481106077,
      "r": 0.2489811697983504
    },
    "rouge-2": {
      "f": 0.09356939220508297,
      "p": 0.09877845706232852,
      "r": 0.09682421744837488
    },
    "rouge-l": {
      "f": 0.22007317044713115,
      "p": 0.23617723049045597,
      "r": 0.2216698935716019
    }
  }
```

