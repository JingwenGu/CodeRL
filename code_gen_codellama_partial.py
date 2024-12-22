import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from utils import llamaInference

split_code_dir = 'split_code_codellama_100_h.json'
with open(split_code_dir,'r') as file:
    split_code = json.load(file)

#token = "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"
#model_path = "meta-llama/Meta-Llama-3.1-8B"
model_path = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()
#convert(model,inplace=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

def llamaInference(prompt):
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        outputs = model.generate(
            input_ids, 
            max_new_tokens=1000, 
            num_return_sequences=1,
            repetition_penalty=1.2,
            temperature=0.7,
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

dataset = 'humaneval'
if dataset == 'apps':
    dataset_dir = 'APPS/test'
    generated_code = {}
    for problem in tqdm.tqdm(os.listdir(dataset_dir)):
        if int(problem) > 100:
            continue
        try:
            problem_path = os.path.join(dataset_dir,problem)
            with open(os.path.join(problem_path,'question.txt'), 'r') as file:
                question = file.read()
            prompt = f'[INST] Write a Python function to solve the following problem:\n\n{question}\n\nReturn the code only, no other explanation is necessary. Make sure that the function is named exactly \"solution\", and make sure that the function is parameterless, but manages input and output as indicated in the problem. [/INST]'
            code = [llamaInference(f'{prompt}{partial_code}') for partial_code in split_code[problem]]
            for c in code:
                print(c)
            generated_code[problem] = list(zip(split_code[problem],code))
        except:
            print('ERROR')
    with open('generated_code_codellama_100_partial_h.json','w') as file:
        json.dump(generated_code,file)

elif dataset == 'humaneval':
    with open('split_code_codellama_humaneval_3-2.json','r') as file:
        problems = json.load(file)
    # with open('evaluated_code_codellama_humaneval_3-2.json','r') as file:
    #     evaluated = json.load(file)
    generated_code = {}
    for key,problem in tqdm.tqdm(problems.items()):
        if problem['canonical_solution'].count('\n') < 7:
            continue
        codes = []
        for i,partial_code in enumerate(problem['code']):
            try:
                code = llamaInference(partial_code)
                print(code)
                codes.append(code)
            except:
                print('ERROR')
        generated_code[key] = {'problem':problem['problem'],'code':codes,'entry_point':problem['entry_point'],'canonical_solution':problem['canonical_solution'],'test':problem['test']}
    with open('generated_code_codellama_humaneval_3-2_partial.json','w') as file:
        json.dump(generated_code,file)  