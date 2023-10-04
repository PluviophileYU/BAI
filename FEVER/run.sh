#!/usr/bin/env bash
for((i=1;i<=7;i++));
do
    python main.py --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --do_train \
        --do_eval \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --max_seq_length 128 \
        --output_dir ../output \
        --per_gpu_eval_batch_size 128  \
        --per_gpu_train_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --eval_all_checkpoints \
        --fp16 \
        --task_name fever \
        --save_steps 1000 \
        --env_file env-2021-10-13-00-18.npy \
        --num_envs 5 \
        --recur_dir N.A \
        --inv_penalty_scale 1e5 \
        --seed $i
done

