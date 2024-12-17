import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel

# from tensorboard import notebook
# log_dir = "results/runs"
# notebook.start("--logdir {} --port 4000".format(log_dir))

new_model = "./hukka_hua_2"
# new_model = "meta-llama/Llama-2-7b-hf"

logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "Get instant approval on your $5000 loan! Just pay a $200 processing fee upfront to get started"
pipe = pipeline(task="text-generation", model=new_model, tokenizer=new_model, max_length=256)
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# result = pipe(f"For the given sentence \"{prompt}\", if some one posts it, will it violate any rule? If violation is true then which rule is violated and what is the explanation")
result = pipe(f"For the given sentence \"{prompt}\", Does it violate any rules of facebook? Answer in yes or no")

print(result[0]['generated_text'])

# prompt = "What is Datacamp Career track?"
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])

# # Reload model in FP16 and merge it with LoRA weights
# load_model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.float16,
#     device_map={"": 0},
# )

# model = PeftModel.from_pretrained(load_model, new_model)
# model = model.merge_and_unload()

# # Reload tokenizer to save it
# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"