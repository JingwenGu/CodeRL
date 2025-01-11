import json
import os
import io
import sys
import tqdm
import multiprocessing as mp
from transformers import AutoTokenizer,HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

def splitCode(problem,code,k,K):
    head = code[:code.find(problem)+len(problem)]
    code = code[code.find(problem)+len(problem):]
    code = code.replace('`','')
    if code.find('if __name__') != -1:
        code = code[:code.find('if __name__')]
    lines = code.split('\n')
    L = (len(lines)-1)//K
    if L == 0:
        return -1
    if k == 0:
        return head #head (problem) should end with a \n
    else:
        chunk = lines[:k*L] if k < K else lines
        return head+'\n'+"\n".join(chunk) + '\n'

@dataclass
class ScriptArguments:
    beam_eval_dir: Optional[str] = field(default="")
    eval_dir: Optional[str] = field(default="")
    result_dir: Optional[str] = field(default="")

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

with open(script_args.beam_eval_dir,'r') as file:
    evaluated_code = json.load(file)
with open(script_args.eval_dir,'r') as file:
    evaluated_code0 = json.load(file)

n_steps = 3
n_branches = 3

model_path = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

selected_code = {}
n_selected = 0
overall = [0,0]
for key,problem in tqdm.tqdm(evaluated_code.items()):
    print(key)
    selected = []
    # iterate through trajectories
    for code,result,result0 in zip(problem['code'],problem['result'],evaluated_code0[key]['result']):
        step_acc = []
        for step,step_result in enumerate(result):
            acc = len([a for a in step_result if a.find('Accepted') != -1])/len(step_result)
            step_acc.append(acc)
        print(step_acc)
        code_ids = tokenizer(code, return_tensors="pt").input_ids
        code_ids = code_ids[0].tolist()
        tokenized_problem = tokenizer(problem['problem'], return_tensors="pt").input_ids
        head_length = tokenized_problem.shape[-1]
        chunk_length = (len(code_ids)-head_length)//n_steps
        for step in range(1,len(step_acc)):
            if (step < n_steps-1 and step_acc[step] < step_acc[step+1]) or (step_acc[step] < 1 and result0.find('Accepted') != -1):
                cutoff = head_length + step*chunk_length if step < n_steps else len(code_ids)
                cutoff0 = head_length + (step-1)*chunk_length
                partial_ids = code_ids[:cutoff]
                selected.append((partial_ids,cutoff0))
    for traj in problem['result']:
        for step in traj:
            for r in step:
                if 'Accepted' in r:
                    overall[1] += 1
                else:
                    overall[0] += 1
    selected_code[key] = problem.copy()
    selected_code[key]['selected'] = selected
    n_selected += len(selected)
print(f'====={n_selected}=====')
print(overall,overall[1]/(overall[0]+overall[1]))
with open(script_args.result_dir,'w') as file:
    json.dump(selected_code,file)