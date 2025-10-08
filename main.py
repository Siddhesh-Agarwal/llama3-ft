#!/usr/bin/env python3
"""
Llama 3 LoRA Fine-tuning Script
No fluff, just working code.
"""

import torch
import typer
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import BitsAndBytesConfig


app = typer.Typer(help="Fine-tune Llama 3 with LoRA. No nonsense.")
console = Console()


def load_jsonl_data(file_path: str):
    """Load JSONL file and return dataset."""
    return load_dataset("json", data_files=file_path, split="train")


def tokenize_function(examples, tokenizer, max_length: int):
    """Tokenize the dataset. Expects 'text' field in JSONL."""
    # Adjust this if your JSONL has different field names
    if "text" in examples:
        texts = examples["text"]
    elif "prompt" in examples and "completion" in examples:
        texts = [
            f"{prompt}\n{completion}"
            for prompt, completion in zip(examples["prompt"], examples["completion"])
        ]
    else:
        raise ValueError(
            "JSONL must contain 'text' field or 'prompt' and 'completion' fields"
        )

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def setup_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Initialize model and tokenizer with quantization."""

    # Quantization config for memory efficiency
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_lora(
    model: PreTrainedModel,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """Configure LoRA adapters."""

    if target_modules is None:
        # Default target modules for Llama 3
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


@app.command()
def main(
    # Data arguments
    data_path: Path = typer.Option(
        ...,
        help="Path to JSONL file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    model_name: str = typer.Option(
        "meta-llama/Meta-Llama-3-8B", help="Base model name"
    ),
    output_dir: Path = typer.Option("./llama3-lora-finetuned", help="Output directory"),
    # LoRA hyperparameters
    lora_r: int = typer.Option(16, help="LoRA rank", min=1, max=256),
    lora_alpha: int = typer.Option(32, help="LoRA alpha", min=1),
    lora_dropout: float = typer.Option(0.05, help="LoRA dropout", min=0.0, max=1.0),
    # Training hyperparameters
    batch_size: int = typer.Option(4, help="Training batch size", min=1),
    gradient_accumulation_steps: int = typer.Option(
        4, help="Gradient accumulation steps", min=1
    ),
    learning_rate: float = typer.Option(2e-4, help="Learning rate", min=1e-6, max=1e-2),
    num_epochs: int = typer.Option(3, help="Number of training epochs", min=1),
    max_length: int = typer.Option(
        512, help="Maximum sequence length", min=1, max=4096
    ),
    warmup_steps: int = typer.Option(100, help="Warmup steps", min=0),
    logging_steps: int = typer.Option(10, help="Logging frequency", min=1),
    save_steps: int = typer.Option(500, help="Save checkpoint frequency", min=1),
    use_4bit: bool = typer.Option(True, help="Use 4-bit quantization"),
):
    """Fine-tune Llama 3 with LoRA."""

    console.rule()
    with console.status("Loading model and tokenizer..."):
        model, tokenizer = setup_model_and_tokenizer(model_name, use_4bit)

    with console.status("Setting up LoRA..."):
        lora_model = setup_lora(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    with console.status("Loading dataset..."):
        dataset = load_jsonl_data(str(data_path))

    console.print(f"Dataset size: {len(dataset)}")
    with console.status("Tokenizing dataset..."):
        tokenized_dataset = dataset.map(
            lambda x: tokenize_function(x, tokenizer, max_length),
            batched=True,
            remove_columns=dataset.column_names,
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        warmup_steps=warmup_steps,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Trainer
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    console.rule()
    with console.status("Starting training..."):
        trainer.train()

    with console.status("Saving model..."):
        lora_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    console.print(f"Model saved to {output_dir}")
    console.print("Done. Now go use it.")


if __name__ == "__main__":
    app()
