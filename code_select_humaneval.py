import json
import os
import io
import sys
import tqdm
import multiprocessing as mp

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
    chunk = ""
    if k == 0:
        chunk = lines[1:L]
    elif k == K-1:
        chunk = lines[k*L:]
    else:
        chunk = lines[k*L:(k+1)*L]
    return head+'\n'+"\n".join(chunk)

with open('code_3_1e.json','r') as file:
    evaluated_code = json.load(file)

n_steps = 3
n_branches = 3

selected_code = {}
n_selected = 0
n_selected0 = 0
overall = [0,0]
for key,problem in tqdm.tqdm(evaluated_code.items()):
    print(key)
    selected = []
    for code in problem['code']:
        step_acc = []
        for k in range(n_steps-1):
            #print(problem['problem'])
            partial_code = splitCode(problem['problem'],code,k,n_steps)
            if partial_code == -1:
                continue
            accuracies = []
            for i,new_codes in enumerate(problem['beams']):
                if new_codes[0].find(partial_code) != -1:
                    acc = len([a for a in problem['result'][i] if a.find('Accepted') != -1])/len(problem['result'][i])
                    accuracies.append(acc)
            if len(accuracies) == 0:
                # print('000')
                continue
            mean_acc = sum(accuracies)/len(accuracies)
            step_acc.append((partial_code,mean_acc))
        for k in range(len(step_acc)-1):
            if step_acc[k][1] <= step_acc[k+1][1] and step_acc[k+1][1]>0:
                selected.append(step_acc[k+1][0])
                print(step_acc[k+1][1])
            if step_acc[k][1] <= step_acc[k+1][1]:
                n_selected0 += 1
    for res in problem['result']:
        for r in res:
            if 'Accepted' in r:
                overall[1] += 1
            else:
                overall[0] += 1
    selected_code[key] = problem.copy()
    selected_code['selected'] = selected
    n_selected += len(selected)
print(f'====={n_selected}=====')
print(f'====={n_selected0}=====')
print(overall,overall[1]/(overall[0]+overall[1]))
# with open('code_3_1s.json','w') as file:
#     json.dump(selected_code,file) 