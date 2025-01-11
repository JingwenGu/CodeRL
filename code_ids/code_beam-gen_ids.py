import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    base_dir: Optional[str] = field(default="")
    n_steps: Optional[int] = field(default=3)
    n_branches: Optional[int] = field(default=3)
    result_dir: Optional[str] = field(default="")

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

with open(script_args.base_dir,'r') as file:
    base_code = json.load(file)

def inferenceTextID_ID(model,tokenizer,device,ids,num_outputs=1):
    with torch.no_grad():
        input_ids = torch.tensor(ids,dtype=torch.int64).unsqueeze(0)
        input_ids = input_ids.to(device)
        print(input_ids)
        print(input_ids.shape)
        outputs = model.generate(
            input_ids, 
            max_new_tokens=1000, 
            num_return_sequences=num_outputs,
            do_sample = True,
            # repetition_penalty=1.2,
            temperature=0.7,
        )
        generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_ids = [output.tolist() for output in outputs]
    return generated_text,generated_ids

def inferenceTextID(prompt,num_outputs=1):
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        print(input_ids)
        print(input_ids.shape)
        outputs = model.generate(
            input_ids, 
            max_new_tokens=1000, 
            num_return_sequences=num_outputs,
            do_sample = True,
            # repetition_penalty=1.2,
            temperature=0.7,
        )
        generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_ids = [output.tolist() for output in outputs]
    return generated_text,generated_ids

n_steps = script_args.n_steps
n_branches = script_args.n_branches

model_path = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generated_code = {}
count = 0
interrupt = False
for key,problem in tqdm.tqdm(base_code.items()):
    if interrupt:
        break
    if problem['canonical_solution'].count('\n') < 7:
        continue
    print(key)
    count += 1
    # if count <= 24:
    #     continue
    beams = []
    for code,code_ids in zip(problem['code'],problem['code_ids']):
        if interrupt:
            break
        steps = []
        tokenized_problem = tokenizer(problem['problem'], return_tensors="pt").input_ids
        head_length = tokenized_problem.shape[-1]
        chunk_length = (len(code_ids)-head_length)//n_steps
        for k in range(n_steps):
            if interrupt:
                break
            # try:
            print(problem['problem'])
            cutoff = head_length + k*chunk_length if k < n_steps else len(code_ids)
            partial_ids = code_ids[:cutoff]
            # inferenceTextID(problem['problem'])
            new_code,new_code_ids = inferenceTextID_ID(model,tokenizer,device,partial_ids,num_outputs=n_branches)
            print(new_code)
            steps.append(new_code)
            # except KeyboardInterrupt:
            #     interrupt = True
            # except:
            #     print('ERROR')
        beams.append(steps)
    generated_code[key] = problem.copy()
    generated_code[key]['beams'] = beams
with open(script_args.result_dir,'w') as file:
    json.dump(generated_code,file) 