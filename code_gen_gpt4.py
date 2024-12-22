import os
import json
import requests
import openai
from openai import OpenAI

# creds = get_openai_creds()
# api_key = creds['openai_key']
# api_org = creds['openai_org']
client = OpenAI(api_key=api_key, organization=api_org)
    
def requestPrompt(prompt,max_tokens=3000,n=1):
  my_api_key = "sk-proj-LUBboD5Qd1xzsehYViPVT3BlbkFJ4CCzyNCsGxI6tpICgOOK"

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {my_api_key}"
  }

  payload = {
      "model": "gpt-4o",
      "messages": [
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": prompt
                  }
              ]
          }
      ],
      "n": n,
      #"max_tokens": max_tokens
  }

  try:
      response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
      print(response.json())
      return response.json()
  except Exception as ex:
      print("Exception:", ex)

dataset_dir = 'APPS/test'
generated_code = {}
for problem in os.listdir(dataset_dir):
    if int(problem) > 100:
        continue
    problem_path = os.path.join(dataset_dir,problem)
    with open(os.path.join(problem_path,'question.txt'), 'r') as file:
        question = file.read()
    code = requestPrompt(f'Write a Python function to solve the following problem:\n\n{question}\n\nReturn the code only, no other explanation is necessary. Make sure that the function is named exactly \"solution\", and make sure that the function is parameterless, but manages input and output as indicated in the problem. Python code:')
    generated_code[problem] = code
with open('generated_code_gpt4_100_1.json','w') as file:
    json.dump(generated_code,file)
