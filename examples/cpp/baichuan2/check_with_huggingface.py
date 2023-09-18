import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/data/llama-7b-hf"
prompt = "Hey, are you consciours? Can you talk to me?"
rtype = 0  # prompt_token 0 python prompt 1 token decoder 2
tokens = [
    0
]

tokenizer = AutoModelForCausalLM.from_pretrained(model_path)
inputs = tokenizer(prompt, return_tensors='pt')
if rtype == 0: print(inputs); exit()
if rtype == 1:
    model = AutoTokenizer.from_pretrained(model_path)
    hf_config = vars(model.config)
    generated_ids = model.forward(inputs.input_ids, output_hidden_states=True)
    print(generated_ids)

if rtype == 2: print(tokenizer.decode(tokens))
