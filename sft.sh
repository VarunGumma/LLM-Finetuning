#!/bin/bash 
HF_TOKEN="<your-hf-token>"
MODEL="microsoft/Phi-3-mini-128k-instruct"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file "ds_configs/stage3_no_offloading_accelerate.conf" \
    sft.py \
    --task sft \
    --do_eval \
    --low_cpu_mem_usage \
    --train_local_dataset "data/sft/train" \
    --eval_local_dataset "data/sft/test" \
    --gradient_checkpointing \
    --model $MODEL \
    --attn_implementation flash_attention_2 \
    --output_dir "output/sft/${MODEL}" \
    --lora_r 256 \
    --lora_alpha 512 \
    --lora_dropout 0.1 \
    --lora_target_modules all-linear \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --optimizer adamw_torch_fused \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --bf16 \
    --tf32 \
    --hf_token $HF_TOKEN \
    --trust_remote_code \
    --report_to wandb \
    --save_total_limit 1 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --dataset_num_proc 64 