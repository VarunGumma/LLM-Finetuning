#!/bin/bash 
MODEL="microsoft/Phi-3-mini-128k-instruct"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file "ds_configs/stage3_no_offloading_accelerate.conf" \
    dpo.py \
    --task dpo \
    --do_eval \
    --train_local_dataset processed_data/dpo/train \
    --eval_local_dataset processed_data/dpo/test \
    --model $MODEL \
    --attn_implementation flash_attention_2 \
    --output_dir "output/dpo/${MODEL}" \
    --lora_r 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target_modules all-linear \
    --gradient_checkpointing \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --optimizer adamw_bnb_8bit \
    --lr_scheduler_type cosine \
    --max_grad_norm 0.3 \
    --bf16 \
    --tf32 \
    --beta 0.1 \
    --loss_type ipo \
    --trust_remote_code \
    --report_to none \
    --save_total_limit 1 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --dataset_num_proc 64