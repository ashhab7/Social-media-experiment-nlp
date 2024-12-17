import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nltk
import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Dataset
df = pd.read_csv(
    '/scratch/ftm2nu/sentiment_analysis/training.1600000.processed.noemoticon.csv',
    encoding='latin-1'
)
df = df[df.iloc[:, 0] != 2]
df.iloc[df.iloc[:, 0] == 4, 0] = 1
train_df, temp_df = train_test_split(df, train_size=0.8, random_state=42)
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Print the shapes of the resulting sets
print("Train set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("Validation set shape:", val_df.shape)

y_train = train_df.iloc[:, 0].values  # Selecting the first column as the label
X_train = train_df.iloc[:, 5].values  # Selecting all columns except the first one as features
y_valid = val_df.iloc[:, 0].values  # Selecting the first column as the label
X_valid = val_df.iloc[:, 5].values  # Selecting all columns except the first one as features
y_test = test_df.iloc[:, 0].values  # Selecting the first column as the label
X_test = test_df.iloc[:, 5].values  # Selecting all columns except the first one as features



class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()  # Using ReLU as the hidden layer activation
        self.softmax = nn.Softmax(dim=1)  # For the output layer

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.softmax(self.fc5(x))
        return x

nltk.download('punkt')
nltk.download('punkt_tab')

def tokenizer(s):
    return word_tokenize(s)



# Function to read JSONL file and convert to DataFrame
def jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    df = pd.json_normalize(data)
    return df

# Usage example
file_path = 'support_converted.jsonl'
new_df = jsonl_to_dataframe(file_path)

# Display the DataFrame

user_input = new_df.iloc[:, 0].values
# user_input = ['I am joyful today']
# print(type(user_input))


# # user_input = ["You are soo dead"]
neural_model = torch.load("chunked_model.pth")
neural_model.eval()

cv_train = CountVectorizer(lowercase=True, tokenizer=tokenizer, min_df=10, max_df=0.9)
train_vec_data = cv_train.fit_transform(X_train)
valid_vec_data = cv_train.transform(user_input)

x_valid_tensor = torch.tensor(valid_vec_data.toarray(), dtype=torch.float32)
outputs = neural_model(x_valid_tensor)
predicted = torch.nn.functional.softmax(outputs, dim = 1).detach().cpu().numpy()


temp_list = []


for i in range(len(user_input)):
    temp_list.append({"input":user_input[i], "pos_prob":predicted[i][0], "neg_prob":predicted[i][1]})
    # print({"input":user_input[i], "pos_prob":predicted[i][0], "neg_prob":predicted[i][1]})
    


# Function to merge JSON objects based on input key and save to JSONL
def merge_json_objects(jsonl_file_1, json_list_2, output_file):
    # Read the first JSONL file
    data_1 = {}
    with open(jsonl_file_1, 'r') as f:
        for line in f:
            obj = json.loads(line.strip())
            data_1[obj['input']] = obj

    # Merge with the second list of JSON objects
    for obj in json_list_2:
        input_key = obj['input']
        new_obj = {
            "input": {
                "post": obj['input'],
                "pos_prob": float(obj['pos_prob']),
                "neg_prob": float(obj['neg_prob'])
            }
        }
        if input_key in data_1:
            data_1[input_key].update(new_obj)
        else:
            data_1[input_key] = new_obj

    # Write merged objects to a new JSONL file
    with open(output_file, 'w') as f:
        for obj in data_1.values():
            # print(obj)
            json.dump(obj, f) 
            f.write('\n')

# Usage example
jsonl_file_1 = 'support_converted.jsonl'

output_file = 'merged_output.jsonl'

merge_json_objects(jsonl_file_1, temp_list, output_file)

print("Merged JSON objects have been saved to", output_file)


# Function to convert JSONL to formatted text
def convert_jsonl_to_text(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            obj = json.loads(line.strip())
            formatted_text = (
                f"<s> [INST] If the given sentence ```{obj['input']['post']}``` is posted in any social media, with positive probability {obj['input']['pos_prob']} and negative probability {obj['input']['neg_prob']}, will it violate any rule? If violation is true then which rule is violated and what is the explanation  [/INST] "
                f" For the given sentence Rule vioaltion = {obj['output']['violation']} $$$ Rule =  {obj['output']['rule']} $$$ Explanation = {obj['output']['explanation']}</s>\n"
            )
            f_out.write(formatted_text)

# Usage example
input_file = 'merged_output.jsonl'
output_file = 'train_output.txt'
convert_jsonl_to_text(input_file, output_file)

print("Formatted text has been saved to", output_file)




# with open("propt.txt", "r") as file:
#     prompt_template = file.read()

# formatted_prompt = prompt_template.format(Post = user_input[0])

# access_token = "give token"

# # Load the Llama-3-8B model and tokenizer
# # mymodel = "meta-llama/Meta-Llama-3-8B"  # Make sure to use the correct model name
# # Load model directly

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")


# # mymodel = "meta-llama/Llama-3.1-8B" 
# # model = AutoModelForCausalLM.from_pretrained(mymodel, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
# # tokenizer = AutoTokenizer.from_pretrained(mymodel, token=access_token, torch_dtype=torch.bfloat16, trust_remote_code=True)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)
# # Encode input text and generate output
# inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

#     # Generate the output (test case)
# outputs = model.generate(
#         **inputs,
#         max_length=300,  # Set the maximum length of the output
#         num_return_sequences=1,  # Generate multiple test cases if needed
#         no_repeat_ngram_size=2,  # Avoid repeating n-grams for diversity
#         temperature=0.7,  # Controls the randomness of the output (higher for more creativity)
#         top_p=0.9,  # Nucleus sampling, adjust to control diversity
#     )

#     # Decode and return the generated text
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(generated_text)
