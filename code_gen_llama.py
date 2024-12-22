import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

token = "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"
#model_path = "meta-llama/Meta-Llama-3.1-8B"
model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, token=token, device_map="auto")
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

dataset_dir = 'APPS/test'
generated_code = {}
for problem in tqdm.tqdm(os.listdir(dataset_dir)):
    if int(problem) > 100:
        continue
    try:
        problem_path = os.path.join(dataset_dir,problem)
        with open(os.path.join(problem_path,'question.txt'), 'r') as file:
            question = file.read()
        code = llamaInference(f'<|USER|> Write a Python function to solve the following problem:\n\n{question}\n\nReturn the code only, no other explanation is necessary. Make sure that the function is named exactly \"solution\", and make sure that the function is parameterless, but manages input and output as indicated in the problem. <|USER|> <|ASSISTANT|>')
        print(code)
        generated_code[problem] = code
    except:
        print('ERROR')
with open('generated_code_llama-I_100.json','w') as file:
    json.dump(generated_code,file)