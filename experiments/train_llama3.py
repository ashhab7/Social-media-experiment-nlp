import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import Dataset


access_token = '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'



# # Function to read data from text file and append to array
# def read_and_append_to_array(input_file):
#     data_array = []
#     with open(input_file, 'r') as file:
#         for line in file:
#             data_array.append(line.strip())
#     return data_array

# # Usage example
# input_file = 'train_output.txt'
# data_array = read_and_append_to_array(input_file)

# # Display the array
# print(data_array)

json_data = []

def convert_jsonl_to_text(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            obj = json.loads(line.strip())
            json_data.append(obj)
            
            # formatted_text = (
            #     f"<s> [INST] {obj['input']['post']} [/INST] "
            #     f" Rule vioaltion = {obj['output']['violation']} $$$ Rule =  {obj['output']['rule']} $$$ Explanation = {obj['output']['explanation']}</s>\n"
            # )
            # f_out.write(formatted_text)



# Usage example
input_file = 'merged_output.jsonl'
output_file = 'train_output.txt'
convert_jsonl_to_text(input_file, output_file)


def preprocess_data(data):
    processed_data = []
    for item in data:
        input_text = f"For the given sentence \"{item['input']['post']}\", if some one posts it, will it violate any rule? If violation is true then which rule is violated and what is the explanation"
        output_text = (
            f"violation: {item['output']['violation']} \n "
            f"rule: {item['output']['rule']} \n "
            f"explanation: {item['output']['explanation']}"
        )
        # print({"prompt": input_text, "completion": output_text})
        processed_data.append({"prompt": input_text, "completion": output_text})
    return processed_data



processed_data = preprocess_data(json_data)

print(f"length = {len(processed_data)}")

# Tokenizer and model setup
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, token = access_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token = access_token)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
# Create a custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized = self.tokenizer(
            item["prompt"],
            text_pair=item["completion"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss calculation
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

train_dataset = CustomDataset(processed_data, tokenizer)


# LoRA configuration
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


# lora_config = LoraConfig(
#     r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
# )


model = get_peft_model(model, lora_config).to("cuda")

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama2-7b-finetuned",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True,
    push_to_hub=False,
)



# training_args = TrainingArguments(
#     output_dir="./hukka_hua_2",
#     num_train_epochs=1,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=1,
#     # optim="paged_adamw_32bit",
#     save_steps=25,
#     logging_steps=25,
#     learning_rate=2e-4,
#     weight_decay=0.001,
#     fp16=False,
#     bf16=False,
#     max_grad_norm=0.3,
#     max_steps=-1,
#     warmup_ratio=0.03,
#     group_by_length=True,
#     lr_scheduler_type="constant",
#     # report_to="tensorboard"
# )

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Fine-tuning
trainer.train()

# Save the fine-tuned model
# model.save("./llama2-7b-finetuned1")
trainer.model.save_pretrained("hukka_hua_2")
trainer.tokenizer.save_pretrained("hukka_hua_2")
