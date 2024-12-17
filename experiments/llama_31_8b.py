import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

access_token = "give_token"

# Load the Llama-3-8B model and tokenizer
# mymodel = "meta-llama/Meta-Llama-3-8B"  # Make sure to use the correct model name
mymodel = "meta-llama/Llama-3.1-8B" 
model = AutoModelForCausalLM.from_pretrained(mymodel, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(mymodel, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True)
# generator = pipeline("text-generation",model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device=0, token=access_token)


# Check if GPU is available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_test_case(prompt: str, max_length: int = 300, num_return_sequences: int = 1):
    # Encode the prompt and move to the appropriate device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate the output (test case)
    outputs = model.generate(
        **inputs,
        max_length=max_length,  # Set the maximum length of the output
        num_return_sequences=num_return_sequences,  # Generate multiple test cases if needed
        no_repeat_ngram_size=2,  # Avoid repeating n-grams for diversity
        temperature=0.7,  # Controls the randomness of the output (higher for more creativity)
        top_p=0.9,  # Nucleus sampling, adjust to control diversity
    )

    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Example prompt for generating test cases
    prompt = "Write a paragraph on you"

    # Generate a test case
    test_case = generate_test_case(prompt)
    print("Generated Text:")
    print(test_case)
