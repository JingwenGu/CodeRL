import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    model_dir: Optional[str] = field(default="")
    dataset: Optional[str] = field(default="human_eval")
    num_trajs: Optional[int] = field(default=3)
    result_dir: Optional[str] = field(default="")

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_dir
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()
#convert(model,inplace=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

def inferenceTextID(prompt,num_outputs=1):
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
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

interrupt = False
dataset = script_args.dataset
if dataset == "apps":
    dataset_dir = 'APPS/test'
    generated_code = {}
    for problem in tqdm.tqdm(os.listdir(dataset_dir)):
        if int(problem) > 100:
            continue
        try:
            problem_path = os.path.join(dataset_dir,problem)
            with open(os.path.join(problem_path,'question.txt'), 'r') as file:
                question = file.read()
            code = llamaInference(f'[INST] Write a Python function to solve the following problem:\n\n{question}\n\nReturn the code only, no other explanation is necessary. Make sure that the function is named exactly \"solution\", and make sure that the function is parameterless, but manages input and output as indicated in the problem. [/INST]')
            print(code)
            generated_code[problem] = code
        except:
            print('ERROR')
    with open('generated_code_codellama_100.json','w') as file:
        json.dump(generated_code,file)

elif dataset == "human_eval":
    with open("../human-eval-v2-20210705.jsonl",'r') as file:
        problems = [json.loads(line) for line in file]
    generated_code = {}
    for problem in tqdm.tqdm(problems):
        if problem['canonical_solution'].count('\n') < 7:
            continue
        if interrupt:
            break
        try:
            question = problem['prompt']
            code,code_ids = inferenceTextID(question,num_outputs=script_args.num_trajs)
            print(code)
            generated_code[problem['task_id']] = {'problem':question,'code':code,'code_ids':code_ids,'entry_point':problem['entry_point'],'canonical_solution':problem['canonical_solution'],'test':problem['test']}
        except KeyboardInterrupt:
            interrupt = True
        except:
            print('ERROR')
    with open(script_args.result_dir,'w') as file:
        json.dump(generated_code,file)  