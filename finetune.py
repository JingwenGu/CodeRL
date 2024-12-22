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

with open('selected_codellama_humaneval_3-2.json','r') as f:
    dict = json.load(f)
data_list = []
for problem in dict.values():
    for t in problem['code']:
        code = t[2]
        if code.find('if __name__') != -1:
            code = code[:code.find('if __name__')]
        data_list.append(code)
data_dict = {"text":data_list}
dataset = Dataset.from_dict(data_dict)

def getTokenizedDataLoader(dataset,tokenizer,column,batch_size,max_length,shuffle=False):
    def tokenize_function(examples):
        return tokenizer(examples[column], return_tensors='pt', padding="max_length", truncation=True, max_length=max_length) # used to be 128
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    dataloader = torch.utils.data.DataLoader(
        tokenized_datasets, 
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )
    return dataloader

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

dataloader = getTokenizedDataLoader(dataset=dataset,
                                tokenizer=tokenizer,
                                column='text',
                                batch_size=2,
                                max_length=512,
                                shuffle=False)

# Setup optimizer
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=5e-5)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

output_dir = 'checkpoints/beam_finetuning_1119'
os.makedirs(output_dir, exist_ok=True)
wandb.init(project="beam_finetune", name="beam_finetuning_1119")

# Training loop
epochs = 3
model.train()

step=0
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} / {epochs}")
    for batch in dataloader:
        # Move batch to the same device as the model
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            save_dir = f"{output_dir}/checkpoint-{epoch}-{step}"
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
        print(f"epoch {epoch}, step {step}, loss={loss.item()}")
        wandb.log({"train_loss": loss.item(),"step": step})
        step += 1
        
        if step > 2000:
            break

print("Training complete!")