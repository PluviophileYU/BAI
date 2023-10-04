#!/usr/bin/env bash
for((i=1;i<=7;i++));
do
    python main.py --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --do_train \
        --do_eval \
        --learning_rate 5e-5 \
        --num_train_epochs 3 \
        --max_seq_length 128 \
        --output_dir ../output \
        --per_gpu_eval_batch_size 128  \
        --per_gpu_train_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --eval_all_checkpoints \
        --fp16 \
        --task_name mnli \
        --save_steps 1500 \
        --env_file env-2021-10-03-11-30.npy \
        --num_envs 2 \
        --recur_dir ../output/2021-09-22-12-11-checkpoint-best-star@5,../output/2021-10-06-12-57-checkpoint-best-star@2,../output/2021-10-06-21-34-checkpoint-best-star@2 \
        --inv_penalty_scale 1e2 \
        --seed $i
done

