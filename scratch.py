import json


with open("code_5_0.json",'r') as file:
    problems = [json.loads(line) for line in file]
    print(len(problems))
    # for k,v in problems[0]['HumanEval/161'].items():
    #     print(k)
    #     print(v)
    print(len(problems[0]['HumanEval/161']['code']))
    for code in problems[0]['HumanEval/161']['code']:
        print(code)
        print('======================================================')
print('#########################################################################')

with open("evaluated_code_codellama_humaneval_3-2.json",'r') as file:
    problems = [json.loads(line) for line in file]
    print(len(problems))
    # for k,v in problems[0]['HumanEval/161'].items():
    #     print(k)
    #     print(v)
    print(len(problems[0]['HumanEval/161']['code']))
    for code in problems[0]['HumanEval/161']['code']:
        print(code)
        print('======================================================')