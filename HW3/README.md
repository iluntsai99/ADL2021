# README for Homework 2 ADL NTU 109 Spring

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

## CS and QA
```shell
# training
python train_CS.py --data_dir <data_dir>
python train_QA.py --data_dir <data_dir>
# inference
bash run.sh <context_path> <data_path> <prediction_path>
```

## Testing steps for myself

`bash download.sh`

`bash run.sh ../HW2/hw2_dataset/dataset/context.json ../HW2/hw2_dataset/dataset/public.json pred_public.json`

`bash run.sh ../HW2/hw2_dataset/dataset/context.json ../HW2/hw2_dataset/dataset/private.json pred_private.json`

`python eval.py ../HW2/hw2_dataset/dataset/public.json pred_public.json score.json  `

**Public score should be:**

```json
{'count': 3526, 'em': 0.840045377197958, 'f1': 0.8956364922520763}
```

