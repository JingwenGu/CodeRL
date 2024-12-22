import json
import os
import io
import sys
import multiprocessing as mp

no_suffix = True

processed = {}
generated_code_dir = 'generated_code_codellama_humaneval_3-2.json'
with open(generated_code_dir,'r') as file:
    generated_code = json.load(file)

for key,problem in generated_code.items():
    codes = problem['code'] if type(problem['code']) == list else [problem['code']]
    prompts = []
    for code in codes:
        if 'gpt' in generated_code_dir:
            code = code['choices'][0]['message']['content']
        head = code[:code.find(problem['problem'])+len(problem['problem'])]
        code = code[code.find(problem['problem'])+len(problem['problem']):]
        code = code.replace('`','')
        if code.find('if __name__') != -1:
            code = code[:code.find('if __name__')]
        lines = code.split('\n')
        L = (len(lines)-1)//3
        chunk1 = "\n".join(lines[1:L])
        chunk2 = "\n".join(lines[L:2*L])
        chunk3 = "\n".join(lines[2*L:])
        if no_suffix:
            options = [[head,""],[f'{head}\n{chunk1}',""],[f'{head}\n{chunk1}\n{chunk2}',""]]
            for option in options:
                prompts.append(f'\n{option[0]}\n')
        else:
            options = [(head,""),(head,chunk3),(head,f'{chunk2}\n{chunk3}'),(f'{head}\n{chunk1}',""),(f'{head}\n{chunk1}',chunk3),(f'{head}\n{chunk1}\n{chunk2}',"")]
            for option in options:
                prompts.append(f'<PRE>\n{option[0]}\n<SUF>\n{option[1]}\n<MID>')
    processed[key] = {'problem':problem['problem'],'code':prompts,'entry_point':problem['entry_point'],'canonical_solution':problem['canonical_solution'],'test':problem['test']}
with open('split_code_codellama_humaneval_3-2.json','w') as file:
    json.dump(processed,file)