import json

# Function to convert JSON to JSONL and add IDs
def json_to_jsonl_with_ids(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    with open(output_file, 'w') as f:
        for i, obj in enumerate(data):
            obj['id'] = i
            f.write(json.dumps(obj) + '\n')

# Usage example
input_file = 'support.json'
output_file = 'support_converted.jsonl'
json_to_jsonl_with_ids(input_file, output_file)
