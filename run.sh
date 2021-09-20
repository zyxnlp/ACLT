#!/usr/bin/env bash
export DATA_DIR='./data/'
export MODEL_DIR='./pretrain_model/'
export OUTPUT_DIR='./output/lap14'

CUDA_VISIBLE_DEVICES=0 python ./examples/run_aclt.py --task_name lap --model_type bert --do_train --do_eval --do_test --do_lower_case --overwrite_output_dir --data_dir $DATA_DIR --model_name_or_path $MODEL_DIR --learning_rate 5e-5 --num_train_epochs 19 --max_seq_length 96 --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 64 --gamma 0.1 --output_dir $OUTPUT_DIR



