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
    parser.add_argument("--use_lora", action="store_true", help="Use PEFT")
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
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize the model to 4-bit"
    )
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
    parser.add_argument(
        "--neftune_noise_alpha",
        type=int,
        default=None,
        help="Neftune noise alpha",
    )
    ## For DPO ##
    parser.add_argument("--beta", type=float, default=0.1, help="Loss Beta")
    parser.add_argument("--loss_type", type=str, default="sigmoid", help="Loss type")
    return parser


########################################################################################################################################################################

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def apply_chat_template(example, tokenizer, task="sft"):
    if task == "sft":
        # this uses the custom chat template for a base model
        # this is not recommended if you are using a IFT'ed model
        example["text"] = tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
        return example

    elif task == "dpo":
        # this is function uses the native chat template of the IFT'ed model
        example["text_chosen"] = tokenizer.apply_chat_template(
            example["chosen"][-1:], tokenize=False
        )
        example["text_rejected"] = tokenizer.apply_chat_template(
            example["rejected"][-1:], tokenize=False
        )
        example["text_prompt"] = tokenizer.apply_chat_template(
            example["chosen"][:-1], tokenize=False
        )
    else:
        raise ValueError(f"Invalid task: {task}")
    return example


def get_current_device():
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map():
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_response_template_ids(tokenizer):
    return tokenizer.encode("\n<|assistant|>", add_special_tokens=False)
