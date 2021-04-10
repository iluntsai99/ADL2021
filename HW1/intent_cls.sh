# bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
# "${1}": path to the testing file.
# "${2}": path to the output predictions.

python3 test_intent.py --test_file "${1}" --ckpt_path ckpt/intent/91377.pt --pred_file "${2}"