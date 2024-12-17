import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from datasets import load_dataset
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
from trl import SFTTrainer
from datasets import Dataset


# # Model from Hugging Face hub
# base_model = "NousResearch/Llama-2-7b-chat-hf"

# # New instruction dataset
# guanaco_dataset = "mlabonne/guanaco-llama2-1k"

# # Fine-tuned model
# new_model = "llama-2-7b-chat-guanaco"

# new_model = "hukka_hua_3"

# # Step 1: Read the file
# file_path = 'train_output.txt'  # Replace with the path to your file
# with open(file_path, 'r') as file:
#     lines = file.readlines()

# # Step 2: Process the lines
# temp = [line.strip() for line in lines if line.strip()]
# data = {'text': temp}

# # Step 3: Create a dataset
# dataset = Dataset.from_dict(data)

# # View the dataset
# print(dataset)


# # dataset = load_dataset(guanaco_dataset, split="train")

# # compute_dtype = getattr(torch, "float16")

# # quant_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_compute_dtype=compute_dtype,
# #     bnb_4bit_use_double_quant=False,
# # )


# # Load base model
# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     # quantization_config=quant_config,
#     # device_map={"": 0}
# )
# # model.config.use_cache = False
# # model.config.pretraining_tp = 1


# # Load LLaMA tokenizer
# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# inputs = tokenizer(dataset[0]["text"], return_tensors="pt").to("cuda")
# outputs = model(**inputs)
# print(f"here = {outputs}")


# # Load LoRA configuration
# peft_args = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
# )


# # Set training parameters
# training_params = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=1,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=1,
#     optim="paged_adamw_32bit",
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
#     report_to="tensorboard"
# )

# # Set supervised fine-tuning parameters
# print(dataset.shape)
# # trainer = SFTTrainer(
# #     model=model,
# #     train_dataset=dataset,
# #     peft_config=peft_args,
# #     max_seq_length=512,
# #     tokenizer=tokenizer,
# #     args=training_params,
# #     packing=False,
# # )
# # When creating the SFTTrainer, specify max_seq_length
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_args,
#     dataset_text_field='text',
#     max_seq_length=512,  # Add a specific max sequence length
#     tokenizer=tokenizer,
#     args=training_params,
#     packing=False,
# )
# # Train model
# try: 
#     trainer.train()
# except Exception as e:
#     print(e)
# # Save trained model
# trainer.model.save_pretrained(new_model)


# from tensorboard import notebook
# log_dir = "results/runs"
# notebook.start("--logdir {} --port 4000".format(log_dir))

# # Ignore warnings
# logging.set_verbosity(logging.CRITICAL)

# # Run text generation pipeline with our next model
# prompt = "Who is Leonardo Da Vinci?"
# pipe = pipeline(task="text-generation", model=new_model, tokenizer=new_model, max_length=200)
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])

# prompt = "What is Datacamp Career track?"
# result = pipe(f"<s>[INST] {prompt} [/INST]")
# print(result[0]['generated_text'])

# # # Reload model in FP16 and merge it with LoRA weights
# # load_model = AutoModelForCausalLM.from_pretrained(
# #     base_model,
# #     low_cpu_mem_usage=True,
# #     return_dict=True,
# #     torch_dtype=torch.float16,
# #     device_map={"": 0},
# # )

# # model = PeftModel.from_pretrained(load_model, new_model)
# # model = model.merge_and_unload()


# # # !huggingface-cli login


# # # Reload tokenizer to save it
# # tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# # tokenizer.pad_token = tokenizer.eos_token
# # tokenizer.padding_side = "right"

# # model.push_to_hub(new_model, use_temp_dir=False)
# # tokenizer.push_to_hub(new_model, use_temp_dir=False)
# Add error handling
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer
from datasets import Dataset

def load_dataset_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    temp = [line.strip() for line in lines if line.strip()]
    return Dataset.from_dict({'text': temp})

def main():
    # Configuration
    base_model = "NousResearch/Llama-2-7b-chat-hf"
    new_model = "llama-2-7b-chat-hukka-hua4"
    file_path = 'train_output.txt'

    try:
        # Load dataset
        dataset = load_dataset_from_file(file_path)
        print(f"Dataset loaded with {len(dataset)} samples")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # Optional: add device_map and quantization if needed
            device_map="cuda:0"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # LoRA configuration
        peft_args = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Training arguments
        training_params = TrainingArguments(
            output_dir="./results",
            num_train_epochs=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=torch.cuda.is_available(),
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            logging_steps=10,
        )

        # Initialize trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_args,
            dataset_text_field='text',
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_params,
            packing=False,
        )

        # Train the model
        trainer.train()

        # Save the model
        trainer.model.save_pretrained(new_model)
        trainer.tokenizer.save_pretrained(new_model)
        print(f"Model saved to {new_model}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

