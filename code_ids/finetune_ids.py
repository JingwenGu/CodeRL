import json
import torch
import random
from datasets import Dataset,concatenate_datasets
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, HfArgumentParser, BitsAndBytesConfig
from datasets import load_dataset
from torch.optim import AdamW
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Optional
import time
import wandb
import os
import numpy as np
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@dataclass
class ScriptArguments:
    selected_dir: Optional[str] = field(default="")
    result_dir: Optional[str] = field(default="")
    n_epochs: Optional[int] = field(default=3)
    save_steps: Optional[int] = field(default=10)
    wandb_project: Optional[str] = field(default="beam_finetune")
    wandb_name: Optional[str] = field(default="")

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_path = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path,                        
                                            quantization_config=BitsAndBytesConfig(
                                                load_in_4bit=True,
                                                bnb_4bit_use_double_quant=True,
                                                bnb_4bit_quant_type="nf4",
                                                bnb_4bit_compute_dtype=torch.bfloat16
                                            ),
                                             device_map="auto")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(script_args.selected_dir,'r') as f:
    selected = json.load(f)
data_list = []
cutoff_list = []
for problem in selected.values():
    for code,cutoff in problem['selected']:
        data_list.append(code)
        cutoff_list.append(cutoff)
data_dict = {"input_ids":data_list,'cutoff':cutoff_list}
tokenized_dataset = Dataset.from_dict(data_dict)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

dataloader = torch.utils.data.DataLoader(
    tokenized_dataset, 
    batch_size=2,
    shuffle=False,
    collate_fn=data_collator
)

# @dataclass
# class ScriptArguments:
#     """
#     The name of the Casual LM model we wish to fine with PPO
#     """

#     # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
#     # models like gpt-neo* models are more suitable.
#     model_name: Optional[str] = field(default="gpt2", metadata={"help": "the student model name"})
#     gguf_name: Optional[str] = field(default="gpt2", metadata={"help": "the gguf file name"})
#     dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
#     mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
#     num_epochs: Optional[int] = field(default=3, metadata={"help": "the nubmer of epohcs"})
#     batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
#     save_steps: Optional[int] = field(default=10, metadata={"help": "# steps to save the model"})
#     log_steps: Optional[int] = field(default=10, metadata={"help": "# steps to log the model"})
#     output_dir: Optional[str] = field(default="", metadata={"help": "n steps to save the model"})
#     wandb_project: Optional[str] = field(default="", metadata={"help": "wandb project name"})
#     wandb_run: Optional[str] = field(default="", metadata={"help": "wandb run name"})

# parser = HfArgumentParser(ScriptArguments)
# script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

# start_time = time.perf_counter()


# Setup optimizer
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

output_dir = script_args.result_dir
os.makedirs(output_dir, exist_ok=True)
wandb.init(project=script_args.wandb_project, name="beam_finetuning_1119")

# Training loop
epochs = script_args.n_epochs
model.train()

step=0
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} / {epochs}")
    for batch in dataloader:
        # Move batch to the same device as the model
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        cutoffs = batch['cutoff']

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        # loss = outputs.loss
        logits = outputs.logits

        # Shift logits and labels for causal language modeling
        shift_logits = logits[:, :-1, :]  # Exclude last token for logits
        shift_labels = input_ids[:, 1:]  # Exclude first token for labels
        # print(shift_logits.shape)
        # print(shift_labels.shape)

        # Flatten the logits and labels for loss computation
        batch_size, seq_len, vocab_size = shift_logits.size()
        # shift_logits = shift_logits.view(-1, vocab_size)  # Shape: (batch_size * seq_len, vocab_size)
        # shift_labels = shift_labels.view(-1)  # Shape: (batch_size * seq_len,)
        shift_logits = torch.reshape(shift_logits, (batch_size * seq_len, vocab_size))
        shift_labels = torch.reshape(shift_labels, (batch_size * seq_len,))

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(shift_logits, shift_labels)  # Shape: (batch_size * seq_len)
        token_loss = token_loss.reshape(batch_size, seq_len)
        mask = torch.arange(seq_len).unsqueeze(0) >= cutoffs.unsqueeze(1)  # Shape: (batch_size, seq_len)
        masked_loss = token_loss * mask.float().to(token_loss.device)  # Zero out ignored tokens
        # print(f'masked_loss:{masked_loss}')
        loss = masked_loss.sum() / mask.sum()

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % script_args.save_steps == 0:
            save_dir = f"{output_dir}/checkpoint-{epoch}-{step}"
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
        print(f"epoch {epoch}, step {step}, loss={loss.item()}")
        wandb.log({"train_loss": loss.item(),"step": step})
        step += 1
        
        if step > 2000:
            break

print("Training complete!")