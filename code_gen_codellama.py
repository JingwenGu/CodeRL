import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

#token = "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"
#model_path = "meta-llama/Meta-Llama-3.1-8B"
#model_path = "codellama/CodeLlama-7b-Instruct-hf"
model_path = "checkpoints/beam_finetuning_1119/checkpoint-0-50"
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()
#convert(model,inplace=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

def llamaInference(prompt,num_outputs=1):
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
    return generated_text

dataset = "human_eval"
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
    with open("human-eval-v2-20210705.jsonl",'r') as file:
        problems = [json.loads(line) for line in file]
    generated_code = {}
    for problem in tqdm.tqdm(problems):
        if problem['canonical_solution'].count('\n') < 7:
            continue
        try:
            question = problem['prompt']
            code = llamaInference(question,num_outputs=5)
            print(code)
            generated_code[problem['task_id']] = {'problem':question,'code':code,'entry_point':problem['entry_point'],'canonical_solution':problem['canonical_solution'],'test':problem['test']}
        except:
            print('ERROR')
    with open('code_5_0.json','w') as file:
        json.dump(generated_code,file)  