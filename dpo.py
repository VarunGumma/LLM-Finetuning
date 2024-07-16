import os
import torch
import datasets
from peft import LoraConfig
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)

from utils import *
import warnings
from trl import DPOTrainer, DPOConfig
from dotenv import load_dotenv

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
warnings.filterwarnings("ignore")
load_dotenv()


def main(args):
    if not args.use_unsloth:
        if args.quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            token=os.environ["HF_TOKEN"],
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            use_cache=False if args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if args.quantize else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=False,
            token=os.environ["HF_TOKEN"],
            trust_remote_code=args.trust_remote_code,
        )
    else:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.quantize,
            token=os.environ["HF_TOKEN"],
            dtype=None,
        )

    if "mistral" in args.model or "Llama-2" in args.model:
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE_MISTRAL
        
    if "gemma" in args.model:
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE_GEMMA

    print(f" | > Model: {model}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    if args.train_hf_dataset is not None:
        print(
            f" | > Loading training data from Hugging Face dataset {args.train_hf_dataset}"
        )
        train_dataset = load_dataset(args.train_hf_dataset, split="train")
        print(f" | > Loaded {len(train_dataset)} training examples")
    elif args.train_local_dataset is not None:
        print(
            f" | > Loading training data from local dataset {args.train_local_dataset}"
        )
        train_dataset = datasets.load_from_disk(args.train_local_dataset)
        print(f" | > Loaded {len(train_dataset)} training examples")
    else:
        raise ValueError("No training data provided")

    current_columns = list(train_dataset.features)

    train_dataset = train_dataset.map(
        apply_chat_template,
        num_proc=args.dataset_num_proc,
        remove_columns=current_columns,
        desc=" | > Formatting comparisons with prompt template",
        fn_kwargs={"tokenizer": tokenizer, "task": args.task},
    ).rename_columns(
        {
            "text_prompt": "prompt",
            "text_chosen": "chosen",
            "text_rejected": "rejected",
        }
    ).shuffle()

    if args.do_eval:
        if args.eval_hf_dataset is not None:
            print(
                f" | > Loading evaluation data from Hugging Face dataset {args.eval_hf_dataset}"
            )
            eval_dataset = load_dataset(args.eval_hf_dataset, split="val")
            print(f" | > Loaded {len(eval_dataset)} evaluation examples")
        elif args.eval_local_dataset is not None:
            print(
                f" | > Loading evaluation data from local dataset {args.eval_local_dataset}"
            )
            eval_dataset = datasets.load_from_disk(args.eval_local_dataset)
            print(f" | > Loaded {len(eval_dataset)} evaluation examples")
        else:
            raise ValueError("No evaluation data provided")

        current_columns = list(eval_dataset.features)

        eval_dataset = eval_dataset.map(
            apply_chat_template,
            num_proc=args.dataset_num_proc,
            remove_columns=current_columns,
            desc=" | > Formatting comparisons with prompt template",
            fn_kwargs={"tokenizer": tokenizer, "task": args.task},
        ).rename_columns(
            {
                "text_prompt": "prompt",
                "text_chosen": "chosen",
                "text_rejected": "rejected",
            }
        )
    else:
        eval_dataset = None

    if args.do_test:
        if args.test_hf_dataset is not None:
            print(
                f" | > Loading test data from Hugging Face dataset {args.test_hf_dataset}"
            )
            test_dataset = load_dataset(args.test_hf_dataset, split="test")
            print(f" | > Loaded {len(test_dataset)} test examples")
        elif args.test_local_dataset is not None:
            print(
                f" | > Loading test data from local dataset {args.test_local_dataset}"
            )
            test_dataset = datasets.load_from_disk(args.test_local_dataset)
            print(f" | > Loaded {len(test_dataset)} test examples")
        else:
            raise ValueError("No test data provided")

        current_columns = list(test_dataset.features)

        test_dataset = test_dataset.map(
            apply_chat_template,
            num_proc=args.dataset_num_proc,
            remove_columns=current_columns,
            desc=" | > Formatting comparisons with prompt template",
            fn_kwargs={"tokenizer": tokenizer, "task": args.task},
        ).rename_columns(
            {
                "text_prompt": "prompt",
                "text_chosen": "chosen",
                "text_rejected": "rejected",
            }
        )
    else:
        test_dataset = None

    if args.lora_target_modules[0] == "all-linear":
        args.lora_target_modules = "all-linear"

    if not args.use_unsloth:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            bias="none",
            r=args.lora_r,
            random_state=3407,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
        peft_config = None

    dpo_args = DPOConfig(
        do_eval=args.do_eval,
        optim=args.optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        evaluation_strategy="no" if not args.do_eval else "steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        tf32=args.tf32,
        report_to=args.report_to,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        max_prompt_length=(args.max_seq_length * 2),
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_seq_length,
    )

    trainer = DPOTrainer(
        args=dpo_args,
        model=model,
        ref_model=None,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  
        tokenizer=tokenizer,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted ...")

    print(" | > Training complete. Saving model ...")
    model.save_pretrained(f"{args.output_dir}/best")

    del model, trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
