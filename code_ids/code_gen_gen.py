import os
import json
import torch
import tqdm
import io
import sys
import tqdm
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    model_dir: Optional[str] = field(default="")
    dataset: Optional[str] = field(default="human_eval")
    num_trajs: Optional[int] = field(default=3)
    n_steps: Optional[int] = field(default=3)
    n_branches: Optional[int] = field(default=3)
    result_dir: Optional[str] = field(default="")

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_dir
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.eval()
#convert(model,inplace=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

def inferenceTextID(prompt,num_outputs=1):
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        outputs = model.generate(
            input_ids, 
            max_new_tokens=1000, 
            num_return_sequences=num_outputs,
            do_sample = True,
            # repetition_penalty=1.2,
            temperature=0.7,
        )
        generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_ids = [output.tolist() for output in outputs]
    return generated_text,generated_ids

def inferenceTextID_ID(model,tokenizer,device,ids,num_outputs=1):
    with torch.no_grad():
        input_ids = torch.tensor(ids,dtype=torch.int64).unsqueeze(0)
        input_ids = input_ids.to(device)
        print(input_ids)
        print(input_ids.shape)
        outputs = model.generate(
            input_ids, 
            max_new_tokens=1000, 
            num_return_sequences=num_outputs,
            do_sample = True,
            # repetition_penalty=1.2,
            temperature=0.7,
        )
        generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        generated_ids = [output.tolist() for output in outputs]
    return generated_text,generated_ids

def run_code_with_io_handling(code, func_name=None, inputs=None, timeout=5):
    """
    Executes code with simulated I/O and ensures each test case completes within the timeout.
    :param code: String of code to execute.
    :param func_name: Name of the function to call (optional).
    :param inputs: List or string for stdin simulation.
    :param expected_output: Expected output to compare against.
    :param timeout: Maximum time (in seconds) allowed for the code to run.
    :return: Result message indicating success, failure, or timeout.
    """
    def _execute_code(queue):
        """Helper function to execute code in a separate process."""
        # Prepare input stream
        input_stream = io.StringIO("\n".join(inputs) if isinstance(inputs, list) else inputs)
        output_stream = io.StringIO()

        # Backup original stdin and stdout
        original_stdin = sys.stdin
        original_stdout = sys.stdout

        try:
            # Redirect stdin and stdout
            sys.stdin = input_stream
            sys.stdout = output_stream

            # Compile and execute the code
            compiled_code = compile(code, "<string>", "exec")
            local_namespace = {}
            exec(compiled_code, local_namespace)

            return_val = None
            # Call the function if provided
            if func_name and func_name in local_namespace:
                #local_namespace[func_name]()
                return_val = local_namespace[func_name]()
                if return_val:
                    print(return_val)

            # Get the captured output
            output = output_stream.getvalue().strip()
            queue.put(output)  # Send the output back via the queue
        except AssertionError as e:
            queue.put(f"Failed: Assertion Error - {e}")
        except Exception as e:
            queue.put(f"Failed: Runtime Error - {e}")

        finally:
            # Restore original stdin and stdout
            sys.stdin = original_stdin
            sys.stdout = original_stdout

    # Use a queue to retrieve output from the process
    queue = mp.Queue()
    process = mp.Process(target=_execute_code, args=(queue,))

    # Start the process and wait for it to finish
    process.start()
    process.join(timeout)

    # If the process is still running after the timeout, terminate it
    if process.is_alive():
        process.terminate()
        process.join()
        return f"Failed: Timeout - Code did not finish within {timeout} seconds."

    # Retrieve the output from the queue
    try:
        output = queue.get_nowait()
    except:
        return "Failed: No output was captured."
    return output

interrupt = False
with open("../human-eval-v2-20210705.jsonl",'r') as file:
    problems = [json.loads(line) for line in file]
generated_code = {}
for problem in tqdm.tqdm(problems):
    if problem['canonical_solution'].count('\n') < 7:
        continue
    if interrupt:
        break
    try:
        question = problem['prompt']
        code,code_ids = inferenceTextID(question,num_outputs=script_args.num_trajs)
        print(code)
        generated_code[problem['task_id']] = {'problem':question,'code':code,'code_ids':code_ids,'entry_point':problem['entry_point'],'canonical_solution':problem['canonical_solution'],'test':problem['test']}
    except KeyboardInterrupt:
        interrupt = True
    except:
        print('ERROR')
with open(f'{script_args.result_dir}.json','w') as file:
    json.dump(generated_code,file)

evaluated = {}
for key,problem in tqdm.tqdm(generated_code.items()):
    codes = problem['code'] if type(problem['code']) == list else [problem['code']]
    results = []
    for code in codes:
        fn = code[code.find('def ')+4:code.find('(')]
        print(fn)

        #code = code[code.find('def '):]
        if code.find('if __name__') != -1:
            code = code[:code.find('if __name__')]

        code += '\n' + problem['test'] + '\n'
        code += f'check({fn})'
        print(code)
        num_tests = problem['test'].count('assert')
        result = run_code_with_io_handling(code, func_name="solution", timeout=0.5*num_tests)
        print(result)
        if result.find('Failed: ') != 0:
            result = 'Accepted'
        results.append(result)
    evaluated[key] = problem.copy()
    evaluated[key]['result'] = results
with open(f'{script_args.result_dir}e.json','w') as file:
    json.dump(evaluated,file)

n_steps = script_args.n_steps
n_branches = script_args.n_branches
beam_code = {}
count = 0
interrupt = False
for key,problem in tqdm.tqdm(evaluated.items()):
    if interrupt:
        break
    if problem['canonical_solution'].count('\n') < 7:
        continue
    print(key)
    count += 1
    # if count <= 24:
    #     continue
    beams = []
    for code,code_ids in zip(problem['code'],problem['code_ids']):
        if interrupt:
            break
        steps = []
        tokenized_problem = tokenizer(problem['problem'], return_tensors="pt").input_ids
        head_length = tokenized_problem.shape[-1]
        chunk_length = (len(code_ids)-head_length)//n_steps
        for k in range(n_steps):
            if interrupt:
                break
            # try:
            print(problem['problem'])
            cutoff = head_length + k*chunk_length if k < n_steps else len(code_ids)
            partial_ids = code_ids[:cutoff]
            # inferenceTextID(problem['problem'])
            new_code,new_code_ids = inferenceTextID_ID(model,tokenizer,device,partial_ids,num_outputs=n_branches)
            print(new_code)
            steps.append(new_code)
            # except KeyboardInterrupt:
            #     interrupt = True
            # except:
            #     print('ERROR')
        beams.append(steps)
    beam_code[key] = problem.copy()
    beam_code[key]['beams'] = beams
with open(f'{script_args.result_dir}g.json','w') as file:
    json.dump(beam_code,file)

beam_evaluated = {}
for key,problem in tqdm.tqdm(beam_code.items()):
    results = []
    for trajectory in problem['beams']:
        traj_results = []
        for step in trajectory:
            step_results = []
            for code in step:
                fn = code[code.find('def ')+4:code.find('(')]
                print(fn)

                #code = code[code.find('def '):]
                if code.find('if __name__') != -1:
                    code = code[:code.find('if __name__')]

                code += '\n' + problem['test'] + '\n'
                code += f'check({fn})'
                print(code)
                num_tests = problem['test'].count('assert')
                result = run_code_with_io_handling(code, func_name="solution", timeout=0.5*num_tests)
                print(result)
                if result.find('Failed: ') != 0:
                    result = 'Accepted'
                step_results.append(result)
            traj_results.append(step_results)
        results.append(traj_results)
    beam_evaluated[key] = problem.copy()
    beam_evaluated[key]['result'] = results
with open(f'{script_args.result_dir}ge.json','w') as file:
    json.dump(beam_evaluated,file)

selected_code = {}
n_selected = 0
overall = [0,0]
for key,problem in tqdm.tqdm(beam_evaluated.items()):
    print(key)
    selected = []
    # iterate through trajectories
    for code,result,result0 in zip(problem['code'],problem['result'],evaluated[key]['result']):
        step_acc = []
        for step,step_result in enumerate(result):
            acc = len([a for a in step_result if a.find('Accepted') != -1])/len(step_result)
            step_acc.append(acc)
        print(step_acc)
        code_ids = tokenizer(code, return_tensors="pt").input_ids
        code_ids = code_ids[0].tolist()
        tokenized_problem = tokenizer(problem['problem'], return_tensors="pt").input_ids
        head_length = tokenized_problem.shape[-1]
        chunk_length = (len(code_ids)-head_length)//n_steps
        for step in range(1,len(step_acc)):
            if (step < n_steps-1 and step_acc[step] < step_acc[step+1]) or (step_acc[step] < 1 and result0.find('Accepted') != -1):
                cutoff = head_length + step*chunk_length if step < n_steps else len(code_ids)
                cutoff0 = head_length + (step-1)*chunk_length
                partial_ids = code_ids[:cutoff]
                selected.append((partial_ids,cutoff0))
    for traj in problem['result']:
        for step in traj:
            for r in step:
                if 'Accepted' in r:
                    overall[1] += 1
                else:
                    overall[0] += 1
    selected_code[key] = problem.copy()
    selected_code[key]['selected'] = selected
    n_selected += len(selected)
print(f'====={n_selected}=====')
print(overall,overall[1]/(overall[0]+overall[1]))
with open(f'{script_args.result_dir}s.json','w') as file:
    json.dump(selected_code,file)