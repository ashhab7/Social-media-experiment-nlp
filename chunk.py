import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

def tokenizer(s):
    return word_tokenize(s)

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


# Define Vectorizer Globally
cv_train = CountVectorizer(lowercase=True, tokenizer=tokenizer, min_df=100, max_df=0.7)
train_vec_data = cv_train.fit_transform(X_train)
valid_vec_data = cv_train.transform(X_valid)
test_vec_data = cv_train.transform(X_test)

# Model Definition
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

# Model Parameters
# input_size = train_vec_data.shape[1]  # Set dynamically after vectorization
# hidden_size = 64  # Adjust as needed
# output_size = 2  # Binary classification (0 or 1)
# lr=0.001

# model = NeuralNet(input_size, hidden_size, output_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr)

# Function for Chunked Training
def train_in_chunks(train_vec_data, y_train,hidden_size,lr, chunk_size=100000, batch_size=1000, num_epochs=20):

    global vectorizer, model
    input_size = train_vec_data.shape[1]  # Set dynamically after vectorization
    # hidden_size = hidden_size  # Adjust as needed
    output_size = 2  # Binary classification (0 or 1)
    # lr=lr

    model = NeuralNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    num_chunks = train_vec_data.shape[0] // chunk_size
    print(f"Training on {num_chunks} chunks of {chunk_size} rows each.")
    
    # file_path = "validation_accuracy.txt"

    # Writing accuracy to the file using fopen-style
    # file = open(file_path, "a")  # fopen equivalent in Python

    # for chunk_idx in tqdm(range(num_chunks)):
    #     # Get chunk
    #     print("Going to x chunk")
    #     x_chunk = train_vec_data[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
    #     y_chunk = y_train[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]

    #     # # Vectorize Chunk
    #     # if chunk_idx == 0:
    #     #     X_vec_chunk = vectorizer.fit_transform(X_chunk)  # Fit on the first chunk
    #     # else:
    #     #     X_vec_chunk = vectorizer.transform(X_chunk)
        
    #     # Convert to tensors
    #     X_tensor = torch.tensor(x_chunk.toarray(), dtype=torch.float32)
    #     y_tensor = torch.tensor(y_chunk, dtype=torch.long)
        
    #     # DataLoader for batching
    #     dataset = TensorDataset(X_tensor, y_tensor)
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
    #     # Training Loop
    #     for epoch in range(num_epochs):
    #         epoch_loss = 0.0
    #         for X_batch, y_batch in tqdm(dataloader, desc=f"Chunk {chunk_idx + 1}, Epoch {epoch + 1}"):
    #             # Forward pass
    #             outputs = model(X_batch)
    #             loss = criterion(outputs, y_batch)
    #             epoch_loss += loss.item()
                
    #             # Backward pass and optimization
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #         print(f"Chunk {chunk_idx + 1}, Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    #         file.write(f"Chunk {chunk_idx + 1}, Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    # file.close()  # Close the file to save changes

        
        # Training Loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for chunk_idx in tqdm(range(num_chunks)):
            # Get chunk
            # print("Going to x chunk")
            x_chunk = train_vec_data[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            y_chunk = y_train[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            # Convert to tensors
            X_tensor = torch.tensor(x_chunk.toarray(), dtype=torch.float32)
            y_tensor = torch.tensor(y_chunk, dtype=torch.long)

            # DataLoader for batching
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
            for X_batch, y_batch in tqdm(dataloader, desc=f"Chunk {chunk_idx + 1}, Epoch {epoch + 1}"):
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                epoch_loss += loss.item()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Chunk {chunk_idx + 1}, Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
            # file.write(f"Chunk {chunk_idx + 1}, Epoch {epoch + 1}, Loss: {epoch_loss:.4f} \n")
    # file.close()  # Close the file to save changes


# Train the model in chunks
h_list = [128,256]
lr_list = [0.1,0.001,0.0001,0.0005]
for i in h_list:
    for j in lr_list:
        train_in_chunks(train_vec_data, y_train,hidden_size = i,lr=j)
        print(f"Testing on hidden layer = {i}, learning rate = {j}")
        torch.save(model, f"chunked_model_2_{i}_{j}.pth")
        print("STart validation")
        with torch.no_grad():
    # Convert validation data to tensor
            x_valid_tensor = torch.tensor(valid_vec_data.toarray(), dtype=torch.float32)
            
            # Forward pass through the model
            outputs = model(x_valid_tensor)
            
            # Get predicted class (argmax over the last dimension)
            _, predicted = torch.max(outputs.data, 1)
            
            # Ensure y_valid is a tensor
            if not isinstance(y_valid, torch.Tensor):
                y_valid = torch.tensor(y_valid)
            
            # Calculate accuracy
            accuracy = (predicted == y_valid).sum().item() / y_valid.size(0)
            print(f"Validation Accuracy for hidden size {i} and learning rate {j} = {accuracy}")
            file_path = "validation_accuracy.txt"
            # Writing accuracy to the file
            with open(file_path, "a") as file:
                file.write(f"lr = {j},hidden size = {i} ,Validation Accuracy: {accuracy}")

# # Validation
# with torch.no_grad():
#     x_valid_tensor = torch.tensor(valid_vec_data.toarray(), dtype=torch.float32)
#     outputs = model(x_valid_tensor)
#     _, predicted = torch.max(outputs.data, 1)
#     accuracy = (predicted == y_valid).sum().item() / y_valid.size(0)
#     print("Validation Accuracy:", accuracy)
#     file_path = "validation_accuracy.txt"
#     # Writing accuracy to the file
#     with open(file_path, "a") as file:
#         file.write(f"Validation Accuracy: {accuracy}")




