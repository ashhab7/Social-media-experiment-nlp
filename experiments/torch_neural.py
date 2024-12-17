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
nltk.download('punkt')
nltk.download('punkt_tab')

def tokenizer(s):
    
    return word_tokenize(s)

df = pd.read_csv('/scratch/snf4za/social_context_nlp/sentiment140/versions/2/training.1600000.processed.noemoticon.csv',
                 encoding='latin-1') # Specify the encoding to 'latin-1'
df = df[df.iloc[:, 0] != 2]
df.iloc[df.iloc[:, 0] == 4, 0] = 1


tdf, vdf = train_test_split(df, train_size=0.5, random_state=42)


train_df, temp_df = train_test_split(tdf, train_size=0.8, random_state=42)

# Split the temporary set into test and validation sets
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

print("Count vectorizer")
cv_train = CountVectorizer(lowercase=True, tokenizer=tokenizer, min_df=10, max_df = .9)
train_vec_data = cv_train.fit_transform(X_train)
valid_vec_data = cv_train.transform(X_valid)
test_vec_data = cv_train.transform(X_test)
print(train_vec_data.shape)
print("Convert to tensor")
X_train = torch.tensor(train_vec_data.toarray(), dtype=torch.float32)
X_valid = torch.tensor(valid_vec_data.toarray(), dtype=torch.float32)
X_test = torch.tensor(test_vec_data.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_valid = torch.tensor(y_valid, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
print(X_train.shape)
# X_train = torch.tensor(train_vec_data, dtype=torch.float32)
# X_valid = torch.tensor(valid_vec_data, dtype=torch.float32)
# X_test = torch.tensor(test_vec_data, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
# y_valid = torch.tensor(y_valid, dtype=torch.long)
# y_test = torch.tensor(y_test, dtype=torch.long)
print("Convert completed")

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out

input_size = X_train.shape[1]
hidden_size = 2
output_size = len(torch.unique(y_train))
print("BUild model")
model = NeuralNet(input_size, hidden_size, output_size)
print("Model built")
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
print("Loss calculated")
# DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
print("Tensor created")
# Training loop
num_epochs = 2
for epoch in range(num_epochs):
    for i, (X_batch, y_batch) in tqdm(enumerate(train_loader)):
        y_batch = y_batch.to("cuda")
        # Forward pass
        outputs = model(X_batch).to("cuda")

        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("STart validation")

# Validation
with torch.no_grad():
    outputs = model(X_valid)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_valid).sum().item() / y_valid.size(0)
    print("Validation Accuracy:", accuracy)

torch.save(model, 'model_full.pth')