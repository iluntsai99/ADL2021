# bash ./run.sh /path/to/context.json /path/to/public.json  /path/to/pred/public.json
# "${1}": path to the context file.
# "${2}": path to the testing file.
# "${3}": path to the output predictions.

python3 inference.py --context_path "${1}" --data_path "${2}" --pred_path "${3}"
