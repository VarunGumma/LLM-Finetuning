import argparse
import torch
from accelerate import Accelerator


def get_arg_parser():
    parser = argparse.ArgumentParser(description="SFT/DPO Training")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["sft", "dpo"],
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Hugging Face model id"
    )
    parser.add_argument(
        "--lora_dir", type=str, default=None, help="Pretrained LoRA adapters directory"
    )
    parser.add_argument("--lora_r", type=int, default=256, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=512, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default="all-linear",
        help="LoRA target modules",
    )
    parser.add_argument("--quantize", action="store_true", help="Quantize the model to 4-bit")
    parser.add_argument("--use_unsloth", action="store_true", help="Use UnSloth models")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Low CPU usage when working with quantized models",
    )
    parser.add_argument(
        "--train_hf_dataset",
        type=str,
        default=None,
        help="Hugging Face dataset id for training",
    )
    parser.add_argument(
        "--train_local_dataset",
        type=str,
        default=None,
        help="Train Data Directory, which can be loaded via datasets.load_from_disk",
    )
    parser.add_argument(
        "--eval_hf_dataset",
        type=str,
        default=None,
        help="Hugging Face dataset id for evaluation",
    )
    parser.add_argument(
        "--eval_local_dataset",
        type=str,
        default=None,
        help="Evaluation Data Directory, which can be loaded via datasets.load_from_disk",
    )
    parser.add_argument(
        "--test_hf_dataset",
        type=str,
        default=None,
        help="Hugging Face dataset id for test",
    )
    parser.add_argument(
        "--test_local_dataset",
        type=str,
        default=None,
        help="Test Data Directory, which can be loaded via datasets.load_from_disk",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adamw_torch", help="Optimizer"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.3, help="Maximum gradient norm"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1, help="warmup ratio for training"
    )
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--do_eval", action="store_true", help="Do evaluation")
    parser.add_argument("--do_test", action="store_true", help="Do test")
    parser.add_argument(
        "--save_total_limit", type=int, default=1, help="Save total limit"
    )
    parser.add_argument(
        "--dataset_num_proc", type=int, default=4, help="Dataset number of processes"
    )
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--tf32", action="store_true", help="Use tf32")
    parser.add_argument("--report_to", type=str, default="none", help="Report to")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="trust remote code"
    )
    parser.add_argument(
        "--hf_token", type=str, required=True, help="Hugging Face token"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=4096, help="Maximum sequence length"
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="eval_loss",
        help="Metric for best model",
    )
    parser.add_argument(
        "--greater_is_better",
        action="store_true",
        help="Greater is better for best model",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )
    ## For DPO ##
    parser.add_argument("--beta", type=float, default=0.1, help="Loss Beta")
    parser.add_argument("--loss_type", type=str, default="sigmoid", help="Loss type")
    return parser


########################################################################################################################################################################

FALLBACK_CHAT_TEMPLATE_MISTRAL = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
FALLBACK_CHAT_TEMPLATE_GEMMA = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"


def to_messages(example, has_system_prompt=True):
    if not has_system_prompt:
        user_content = f"{example['instruction']}\n\n{example['input']}"
        assistant_content = example["output"]
        messages = [
            {"content": user_content, "role": "user"},
            {"content": assistant_content, "role": "assistant"},
        ]
    else:
        system_content = example["instruction"]
        user_content = example["input"]
        assistant_content = example["output"]
        messages = [
            {"content": system_content, "role": "system"},
            {"content": user_content, "role": "user"},
            {"content": assistant_content, "role": "assistant"},
        ]
    example["messages"] = messages
    return example


def apply_chat_template(example, tokenizer, task="sft"):
    if task == "sft":
        example["text"] = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    elif task == "dpo":
        prompt_messages = example["chosen"][:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]

        example["text_chosen"] = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False
        )
        example["text_rejected"] = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False
        )
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False
        )
    else:
        raise ValueError(f"Invalid task: {task}")
    return example


def get_current_device():
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map():
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_response_template_ids(tokenizer, model_name):

    if "Phi-3" in model_name:
        response_template_context = "<|assistant|>\n"
    elif "gemma" in model_name:
        response_template_context = "<start_of_turn>model\n"
    elif "Llama-3" in model_name:
        response_template_context = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        response_template_context = "[/INST]"

    response_template_ids = tokenizer.encode(
        response_template_context, add_special_tokens=False
    )

    return response_template_ids
