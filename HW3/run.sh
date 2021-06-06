# bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
# ${1}: path to the input file
# ${2}: path to the output file

python3 inference.py --data_path "${1}" --pred_path "${2}"
