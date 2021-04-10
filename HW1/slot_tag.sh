# bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
# "${1}": path to the testing file.
# "${2}": path to the output predictions.

python3 test_slot.py --test_file "${1}" --ckpt_path ckpt/slot/78552.pt --pred_file "${2}"