#!/bin/bash
#SBATCH -o logs/H3_b4_e2_a3_H4_epoch8_6985.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J H3_b4_e2_a3_H4_epoch8_6985        # 任务名称
#SBATCH --gres=gpu:V100:1
#SBATCH -c 10
CUDA_VISIBLE_DEVICES=0 python3 -u baselines/DMBERT/run_ee.py --data_dir data --model_type bert --model_name_or_path bert-base-uncased --task_name maven --output_dir res --max_seq_length 128 --do_lower_case --per_gpu_train_batch_size 42 --per_gpu_eval_batch_size 42 --gradient_accumulation_steps 3 --learning_rate 4e-5 --num_train_epochs 6 --save_steps 500 --logging_steps 500 --seed 42 --do_test --do_eval --do_train --evaluate_during_training --overwrite_output_dir
CUDA_VISIBLE_DEVICES=0 python3 -u baselines/DMBERT/run_ee.py --data_dir data --model_type bert --model_name_or_path res/checkpoint-5000 --task_name maven_infer --output_dir res --max_seq_length 128 --do_lower_case --per_gpu_train_batch_size 42 --per_gpu_eval_batch_size 42 --gradient_accumulation_steps 2 --learning_rate 4e-5 --num_train_epochs 6 --save_steps 500 --logging_steps 500 --seed 2 --do_infer
CUDA_VISIBLE_DEVICES=0 python3 -u baselines/DMBERT/get_submission.py --test_data data/test.jsonl --preds res/_preds.npy --output upload_res/results.jsonl

