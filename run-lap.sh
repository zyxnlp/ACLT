#!/usr/bin/env bash
export DATA_DIR='./data/'
export MODEL_DIR='./pretrain_model/'
export OUTPUT_DIR='./output/lap14'

TASK='lap'
TYPE='bert'


gamma=0.2
epoch=3
lr=5e-5
device=2


CUDA_VISIBLE_DEVICES=$device python ./examples/run_aclt.py \
--task_name $TASK \
--model_type $TYPE \
--do_train \
--do_eval \
--do_test \
--do_lower_case \
--overwrite_output_dir \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_DIR \
--learning_rate $lr \
--num_train_epochs $epoch \
--max_seq_length 96 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--gamma $gamma \
--output_dir $OUTPUT_DIR

