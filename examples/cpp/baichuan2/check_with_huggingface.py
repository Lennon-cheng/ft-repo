import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/chengxiaojie/models/baichuan2/Baichuan2-13B-Chat"
prompt = "北京有什么著名景点?"
rtype = 2  # prompt_token 0 python prompt 1 token decoder 2
tokens = [
        2420, 5817, 9350, 13824, 74, 2, 2,
]

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors='pt')
if rtype == 0: print(inputs); exit()
if rtype == 1:
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    hf_config = vars(model.config)
    generated_ids = model.forward(inputs.input_ids, output_hidden_states=True)
    print(generated_ids)

if rtype == 2: print(tokenizer.decode(tokens))

