# import json
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
# from peft import LoraConfig, get_peft_model
# import os
access_token = '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'

# # Load the training data
# data = [
#     {
#         "input": "We can completely stop all your debt collections and lawsuits with just a one-time payment.",
#         "output": {
#             "violation": True,
#             "rule": "Debt Relief and Credit Repair Fraud and Scam. Content that: Promises to stop or delete all debt collections or lawsuits.",
#             "explanation": "The example violates the rule as it guarantees the complete cessation of all debt collections and lawsuits, which is often a fraudulent claim."
#         }
#     },
#     {
#         "input": "No more calls from debt collectors! Pay us a fee and we'll make all your debts disappear.",
#         "output": {
#             "violation": True,
#             "rule": "Debt Relief and Credit Repair Fraud and Scam. Content that: Promises to stop or delete all debt collections or lawsuits.",
#             "explanation": "The example violates the rule by promising to make all debts disappear without any legitimate process, which is typically indicative of a scam."
#         }
#     },
#     {
#         "input": "Guaranteed removal of all lawsuits and debt collections from your credit report with our services.",
#         "output": {
#             "violation": True,
#             "rule": "Debt Relief and Credit Repair Fraud and Scam. Content that: Promises to stop or delete all debt collections or lawsuits.",
#             "explanation": "The example violates the rule by making promises to eliminate all lawsuits and debt collections, which is often a fraudulent offer."
#         }
#     }
# ]

# # Preprocess the data for fine-tuning
# def preprocess_data(data):
#     processed_data = []
#     for item in data:
#         input_text = item["input"]
#         output_text = (
#             f"violation: {item['output']['violation']}\n"
#             f"rule: {item['output']['rule']}\n"
#             f"explanation: {item['output']['explanation']}"
#         )
#         processed_data.append({"prompt": input_text, "completion": output_text})
#     return processed_data

# processed_data = preprocess_data(data)

# # Tokenizer and model setup
# print(access_token)
# model_name = "meta-llama/Llama-2-7b-hf"
# model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token).to("cuda")
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# # Tokenize the data
# def tokenize_data(data, tokenizer):
#     inputs = []
#     labels = []
#     for item in data:
#         tokenized = tokenizer(
#             item["prompt"],
#             text_pair=item["completion"],
#             max_length=512,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )
#         inputs.append(tokenized["input_ids"])
#         labels.append(tokenized["input_ids"])
#     return inputs, labels

# inputs, labels = tokenize_data(processed_data, tokenizer)

# # LoRA configuration
# lora_config = LoraConfig(
#     r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
# )
# model = get_peft_model(model, lora_config)

# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./llama2-7b-finetuned",
#     evaluation_strategy="no",
#     learning_rate=2e-5,
#     per_device_train_batch_size=2,
#     num_train_epochs=3,
#     save_strategy="epoch",
#     logging_dir="./logs",
#     fp16=True,
#     push_to_hub=False,
# )

# # Trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=inputs,
#     tokenizer=tokenizer,
# )

# # Fine-tuning
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained("./llama2-7b-finetuned")
# tokenizer.save_pretrained("./llama2-7b-finetuned")

import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import Dataset

# Load the training data
data = [
    {
        "input": "We can completely stop all your debt collections and lawsuits with just a one-time payment.",
        "output": {
            "violation": True,
            "rule": "Debt Relief and Credit Repair Fraud and Scam. Content that: Promises to stop or delete all debt collections or lawsuits.",
            "explanation": "The example violates the rule as it guarantees the complete cessation of all debt collections and lawsuits, which is often a fraudulent claim."
        }
    },
    {
        "input": "No more calls from debt collectors! Pay us a fee and we'll make all your debts disappear.",
        "output": {
            "violation": True,
            "rule": "Debt Relief and Credit Repair Fraud and Scam. Content that: Promises to stop or delete all debt collections or lawsuits.",
            "explanation": "The example violates the rule by promising to make all debts disappear without any legitimate process, which is typically indicative of a scam."
        }
    },
    {
        "input": "Guaranteed removal of all lawsuits and debt collections from your credit report with our services.",
        "output": {
            "violation": True,
            "rule": "Debt Relief and Credit Repair Fraud and Scam. Content that: Promises to stop or delete all debt collections or lawsuits.",
            "explanation": "The example violates the rule by making promises to eliminate all lawsuits and debt collections, which is often a fraudulent offer."
        }
    }
]

# Preprocess the data for fine-tuning
def preprocess_data(data):
    processed_data = []
    for item in data:
        input_text = item["input"]
        output_text = (
            f"violation: {item['output']['violation']}\n"
            f"rule: {item['output']['rule']}\n"
            f"explanation: {item['output']['explanation']}"
        )
        processed_data.append({"prompt": input_text, "completion": output_text})
    return processed_data

processed_data = preprocess_data(data)

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
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
)
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
# model.save("./llama2-7b-finetuned")
# tokenizer.save("./llama2-7b-finetuned")
trainer.model.save_pretrained("hukka_hua")