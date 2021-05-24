#!/bin/bash
source /data/medioli/env/bin/activate
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=true
#export OMP_NUM_THREADS=128
export CUDA_VISIBLE_DEVICES=0

echo "[Training ${1} outputdir /data/medioli/models/mlm/${1}_${3}_${5}_${6}/ with tokenizer config ${2} on dataset ${3} with config ${4} for ${5} epochs]"
nohup python3 -m torch.distributed.launch \
--nproc_per_node 1 run_mlm.py \
--dataset_name $3 \
--tokenizer_name $2 \
--model_type $1 \
--dataset_config_name $4 \
--do_train \
--do_eval \
--learning_rate 1e-5 \
--num_train_epochs $5 \
--save_steps 5000 \
--output_dir /data/medioli/models/mlm/${1}_${3}_${5}_${6} \
--use_fast_tokenizer \
--logging_dir /data/medioli/models/mlm/${1}_${3}_${5}_${6}/runs \
--cache_dir /data/medioli/models/mlm/${1}_${3}_${5}_${6}/cache \
--max_seq_length 512 \
--line_by_line \
--overwrite_output_dir \
--report_to tensorboard &
