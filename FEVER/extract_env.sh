#!/usr/bin/env bash
python main.py --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --do_ref \
        --max_seq_length 128 \
        --output_dir ../output \
        --per_gpu_train_batch_size 512 \
        --gradient_accumulation_steps 1 \
        --fp16 \
        --task_name fever \
        --num_envs 5 \
        --ref_dir ../output/2021-10-13-00-14-checkpoint-best \
        --ref_learning_rate 0.01 \
        --ref_train_steps 1000