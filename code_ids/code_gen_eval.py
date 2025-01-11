import os
import json
import torch
import tqdm
import json
import io
import sys
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ScriptArguments:
    model_dir: Optional[str] = field(default="")
    dataset: Optional[str] = field(default="human_eval")
    num_trajs: Optional[int] = field(default=3)
    result_dir: Optional[str] = field(default="")

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_dir
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inferenceTextID(current_model,prompt,num_outputs=1):
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        outputs = current_model.generate(
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

# checkpoints = [(1,80),(2,120),(4,200),(6,280),(9,400),(11,480),(14,600),(16,680),(19,800)]
checkpoints = [(1,50),(2,100),(3,150),(4,200),(5,250),(7,300),(8,350),(9,400)]
eval_log = []
for c1,c2 in checkpoints:

    model = AutoModelForCausalLM.from_pretrained(f'{model_path}/checkpoint-{c1}-{c2}', device_map="auto")
    model.eval()

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
            code,code_ids = inferenceTextID(model,question,num_outputs=script_args.num_trajs)
            print(code)
            generated_code[problem['task_id']] = {'problem':question,'code':code,'code_ids':code_ids,'entry_point':problem['entry_point'],'canonical_solution':problem['canonical_solution'],'test':problem['test']}
        except KeyboardInterrupt:
            interrupt = True
        except:
            print('ERROR')
    with open(f'{script_args.result_dir}-{c2}.json','w') as file:
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
    with open(f'{script_args.result_dir}-{c2}e.json','w') as file:
        json.dump(evaluated,file)
    
    acc = [0,0]
    count = 0
    for problem in evaluated.values():
        if len(problem['canonical_solution'].split('\n')) <= 7:
            continue
        count += 1
        acc[1] += len(problem['result'])
        for res in problem['result']:
            if res.find('Accepted') != -1:
                acc[0] += 1
    eval_log.append(f'{script_args.result_dir}-{c2}')
    eval_log.append(f'{count} problems with canonical solution length greater than 7')
    eval_log.append(f'{acc[0]} accepted, {acc[1]} failed')
    eval_log.append(f'accuracy={acc[0]/acc[1]}')

    with open(f'{script_args.result_dir}_log.json','w') as file:
        json.dump(eval_log,file)