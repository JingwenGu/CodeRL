import json
import os
import io
import sys
import multiprocessing as mp

no_suffix = True

processed = {}
#generated_code_dir = 'generated_code_gpt4_100_1.json'
#generated_code_dir = 'generated_code_llama-I_100.json'
generated_code_dir = 'generated_code_codellama_100.json'
with open(generated_code_dir,'r') as file:
    generated_code = json.load(file)

dataset_dir = 'APPS/test'
for problem in os.listdir(dataset_dir):
    if not problem in generated_code:
        continue
    print(problem)
    problem_path = os.path.join(dataset_dir,problem)
    if 'gpt' in generated_code_dir:
        text = generated_code[problem]['choices'][0]['message']['content']
    else:
        text = generated_code[problem]
    code = text[text.find('def solution'):]
    code = code.replace('`','')
    code = code[:code.find('if __name__')]
    lines = code.split('\n')
    L = (len(lines)-1)//3
    head = lines[0] + '\n'
    chunk1 = "\n".join(lines[1:L])
    chunk2 = "\n".join(lines[L:2*L])
    chunk3 = "\n".join(lines[2*L:])
    prompts = []
    if no_suffix:
        options = [[head,""],[f'{head}\n{chunk1}',""],[f'{head}\n{chunk1}\n{chunk2}',""]]
        for option in options:
            prompts.append(f'[PYTHON]\n{option[0]}\n')
    else:
        options = [(head,""),(head,chunk3),(head,f'{chunk2}\n{chunk3}'),(f'{head}\n{chunk1}',""),(f'{head}\n{chunk1}',chunk3),(f'{head}\n{chunk1}\n{chunk2}',"")]
        for option in options:
            option1 = option[1] if '[/PYTHON]' in option[1] else option[1] + '\n[/PYTHON]'
            prompts.append(f'[PYTHON] <PRE>\n{option[0]}\n<SUF>\n{option1}\n<MID>')
            #print(prompts[-1])
    processed[problem] = prompts
with open('split_code_codellama_100_h.json','w') as file:
    json.dump(processed,file)