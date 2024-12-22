import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from utils import llamaInference

base_code_dir = 'evaluated_code_codellama_humaneval_3-2.json'
with open(base_code_dir,'r') as file:
    base_code = json.load(file)

n_steps = 3
n_branches = 3

model_path = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def splitCode(problem,code,k,K):
    head = code[:code.find(problem)+len(problem)]
    code = code[code.find(problem)+len(problem):]
    code = code.replace('`','')
    if code.find('if __name__') != -1:
        code = code[:code.find('if __name__')]
    lines = code.split('\n')
    L = (len(lines)-1)//K
    chunk = ""
    if k == 0:
        chunk = lines[1:L]
    elif k == K-1:
        chunk = lines[k*L:]
    else:
        chunk = lines[k*L:(k+1)*L]
    return head+'\n'+"\n".join(chunk)

generated_code = {}
count = 0
for key,problem in tqdm.tqdm(base_code.items()):
    if problem['canonical_solution'].count('\n') < 7:
        continue
    count += 1
    if count <= 29:
        continue
    beam_codes = [[([i],code) for i,code in enumerate(problem['code'])]]
    for k in range(n_steps):
        print(f'{k}===============================')
        curr_layer = []
        #print(type(beam_codes[-1]),type(beam_codes[-1][-1]))
        for identifier,code in beam_codes[-1]:
            print(identifier)
            try:
                print(problem['problem'])
                partial_code = splitCode(problem['problem'],code,k,n_steps)
                print(partial_code)
                N = 1 if k == n_steps-1 else n_branches
                print(N)
                code = llamaInference(model,tokenizer,device,partial_code,num_outputs=N)
                print(code)
                curr_layer.extend([(identifier[:]+[i],code[i]) for i in range(N)])
            except:
                print('ERROR')
            #break
        beam_codes.append(curr_layer)
        #break
    generated_code[key] = {'problem':problem['problem'],'code':beam_codes,'entry_point':problem['entry_point'],'canonical_solution':problem['canonical_solution'],'test':problem['test']}
    with open(f'beam_code_codellama_humaneval_3-2_{count}.json','w') as file:
        json.dump(generated_code,file) 
    #break