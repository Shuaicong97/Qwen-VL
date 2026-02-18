#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
submission_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/utils/ovis_val_preds.jsonl
gt_path=/home/stud/shuaicong/SVAG-Bench/Qwen-VL/data/ovis_val_release.jsonl
save_path=standalone_eval/ovis_val_preds_metrics_r5.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}
