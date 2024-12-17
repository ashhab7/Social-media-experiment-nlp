import pandas as pd
import json

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
df = jsonl_to_dataframe(file_path)

# Display the DataFrame

user_input = df.iloc[:, 0].values
print(type(user_input))