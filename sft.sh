MODEL="TinyLlama/TinyLlama_v1.1"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file "ds_configs/stage3_no_offloading_accelerate.conf" \
    sft.py \
    --task sft \
    --do_eval \
    --train_local_dataset "processed_data/sft/train" \
    --eval_local_dataset "processed_data/sft/test" \
    --model $MODEL \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --output_dir "output/sft/${MODEL}" \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --optimizer adamw_torch \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --neftune_noise_alpha 5 \
    --bf16 \
    --tf32 \
    --trust_remote_code \
    --report_to wandb \
    --save_total_limit 1 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --dataset_num_proc 64 