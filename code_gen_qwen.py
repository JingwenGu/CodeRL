import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

model_path = "Qwen/CodeQwen1.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def QWenInference(prompt):
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

dataset = "apps"
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
            code = QWenInference(f'<|im_start|>user\nWrite a Python function to solve the following problem:\n\n{question}\n\nReturn the code only, no other explanation is necessary. Your entire solution should be encapsulated in the function, and write nothing else. Make sure that the function is named exactly \"solution\", and make sure that the function is parameterless, but manages input and output as indicated in the problem.<|im_end|>\n<|im_start|>assistant\n')
            print(code)
            generated_code[problem] = code
        except:
            print('ERROR')
        break
    # with open('generated_code_qwen_100.json','w') as file:
    #     json.dump(generated_code,file)
elif dataset == "human_eval":
    with open("human-eval-v2-20210705.jsonl",'r') as file:
        problems = [json.loads(line) for line in file]
    generated_code = {}
    for problem in problems:
        question = problem['prompt']
