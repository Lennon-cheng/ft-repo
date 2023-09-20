import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/chengxiaojie/models/baichuan2/Baichuan2-13B-Chat"
prompt = "北京有什么著名景点?"
rtype = 2  # prompt_token 0 python prompt 1 token decoder 2
tokens = [
2420, 5817, 9350, 13824, 74, 100392, 5342, 25220, 92345, 92311, 92336, 72, 92311, 21395, 92311, 92338, 72, 92311, 78729, 92840, 92311, 92354, 72, 92311, 15480, 92311, 92369, 72, 92311, 78341, 92311, 92358, 72, 100170, 93922, 92311, 92373, 72, 92311, 19715, 4731, 92311, 
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

