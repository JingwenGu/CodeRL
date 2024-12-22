import os
import json

base_code_dir = 'beam_evaluated_code_codellama_humaneval_3-2.json'
with open(base_code_dir,'r') as file:
    base_code = json.load(file)

n_steps = 3
n_branches = 3

selected_code = {}
count = 0
for key,problem in base_code.items():
    if problem['canonical_solution'].count('\n') < 7:
        continue
    count += 1
    selected = []
    for k,layer in enumerate(problem['code']):
        if k >= len(problem['code'])-1:
            break
        posterior_mean_accuracy = []
        for identifier,code in layer:
            N = 1 if k == n_steps-1 else n_branches
            acc = 0
            for id,c in problem['result'][k]:
                if id == identifier:
                    acc += 1 if c == 'Accepted' else 0
                    break
            for n in range(N):
                new_identifier = identifier[:]+[n]
                for id,c in problem['result'][k+1]:
                    if id == new_identifier:
                        acc += 1 if c == 'Accepted' else 0
                        break
            acc /= (N+1)
            posterior_mean_accuracy.append((acc,code,identifier))
        threshold_acc = 0
        for acc,code,id in posterior_mean_accuracy:
            threshold_acc += acc
        threshold_acc /= len(posterior_mean_accuracy)
        selected.extend([(id,acc,code) for acc,code,id in posterior_mean_accuracy if acc >= threshold_acc and acc > 0])
    selected_code[key] = {'problem':problem['problem'],'code':selected,'entry_point':problem['entry_point'],'canonical_solution':problem['canonical_solution'],'test':problem['test']}
with open(f'selected_codellama_humaneval_3-2.json','w') as file:
    json.dump(selected_code,file)