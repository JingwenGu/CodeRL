import json
import os
import io
import sys
import tqdm
import multiprocessing as mp

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

evaluated = {}
#generated_code_dir = 'generated_code_gpt4_100_1.json'
#generated_code_dir = 'generated_code_llama-I_100.json'
#generated_code_dir = 'generated_code_codellama_100.json'
generated_code_dir = 'generated_code_codellama_100_partial_h.json'
with open(generated_code_dir,'r') as file:
    generated_code = json.load(file)

dataset_dir = 'APPS/test'
for problem in tqdm.tqdm(os.listdir(dataset_dir)):
    if not problem in generated_code:
        continue
    problem_path = os.path.join(dataset_dir,problem)
    evaluated[problem] = []
    for generation in generated_code[problem]:
        if 'gpt' in generated_code_dir:
            text = generation[1]['choices'][0]['message']['content']
        else:
            text = generation[1]
        #print(text)
        code = text[text.find('def solution'):]
        code = code.replace('`','')
        if code.find('if __name__') != -1:  
            code = code[:code.find('if __name__')]
        lines = code.split('\n')
        imports = """
import math
import sys
from collections import deque
"""
        code = imports + lines[0] + '\n'
        for line in lines[1:]:
            if not line or line[0] == ' ':
                code += line + '\n'
            else:
                break
        print(problem)
        print(code)
        with open(os.path.join(problem_path,'input_output.json'), 'r') as file:
            data = json.load(file)
            inputs = [(x[:-1] if x[-1] == '\n' else x) for x in data['inputs']]
            outputs = [(y[:-1] if y[-1] == '\n' else y) for y in data['outputs']]
            results = [run_code_with_io_handling(code, func_name="solution", inputs=x.split('\n'), timeout=0.5) for x in inputs]
            results = [r.rstrip('\n') for r in results]
            accuracy = sum([(1 if str(r) == y else 0) for (r,y) in zip(results,outputs)])
            print(results)
            print(f"accuracy:{accuracy}/{len(inputs)}={accuracy/len(inputs)}")
            evaluated[problem].append({'input':generation[0],'code':code,'results':results,'accuracy':accuracy/len(inputs)})
with open('evaluated_code_codellama_100_partial_h.json','w') as file:
    json.dump(evaluated,file)