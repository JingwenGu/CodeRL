import torch

def llamaInference(model,tokenizer,device,prompt,num_outputs=1):
    # print("llamaInference")
    # print(device)
    # print(prompt)
    # print(num_outputs)
    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        # print("generating output")
        outputs = model.generate(
            input_ids, 
            max_new_tokens=1000, 
            num_return_sequences=num_outputs,
            do_sample = True,
            # repetition_penalty=1.2,
            temperature=0.7,
        )
        generated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_text