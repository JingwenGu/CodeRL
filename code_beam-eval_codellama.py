import json
import os
import io
import sys
import tqdm
import multiprocessing as mp
#from code_gen_gpt4 import foo

def run_code_with_io_handling(code, func_name=None, inputs=None):
    """
    Executes code with simulated input/output handling.
    :param code: String of code to execute.
    :param func_name: Name of the function to call (optional).
    :param inputs: List of inputs to simulate user input via stdin.
    :return: Captured output or error message.
    """
    # Prepare input and output streams
    input_stream = io.StringIO("\n".join(inputs) if inputs else "")
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

        # If a function name is provided, call it
        if func_name and func_name in local_namespace:
            local_namespace[func_name]()  # Call the function

        # Get the captured output
        output = output_stream.getvalue()
        return output

    except SyntaxError as se:
        return f"Failed: Syntax Error - {se}"

    except Exception as e:
        return f"Failed: Runtime Error - {e}"

    finally:
        # Restore original stdin and stdout
        sys.stdin = original_stdin
        sys.stdout = original_stdout

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

generated_code = {}
idx = [17,22,28,29,31]
for i in idx:
    generated_code_dir = f'beam_code_codellama_humaneval_3-2_{i}.json'
    with open(generated_code_dir,'r') as file:
        generated = json.load(file)
        for k,v in generated.items():
            generated_code[k] = v

evaluated = {}
for key,problem in tqdm.tqdm(generated_code.items()):
    results = []
    for layer in problem['code']:
        layer_results = []
        for identifier,code in layer:
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
            layer_results.append([identifier,result])
        results.append(layer_results)
    evaluated[key] = problem.copy()
    evaluated[key]['result'] = results
with open('beam_evaluated_code_codellama_humaneval_3-2.json','w') as file:
    json.dump(evaluated,file)