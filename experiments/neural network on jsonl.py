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
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Tokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

def tokenizer(s):
    return word_tokenize(s)

# Load Dataset
df = pd.read_csv(
    '/scratch/snf4za/social_context_nlp/sentiment140/versions/2/training.1600000.processed.noemoticon.csv',
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
cv_train = CountVectorizer(lowercase=True, tokenizer=tokenizer, min_df=10, max_df=0.9)
train_vec_data = cv_train.fit_transform(X_train)
valid_vec_data = cv_train.transform(X_valid)
test_vec_data = cv_train.transform(X_test)
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
model = torch.load("chunked_model.pth")
model.eval()

with torch.no_grad():
    # Convert validation data to tensors
    x_valid_tensor = torch.tensor(valid_vec_data.toarray(), dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.long)
    
    # Get model predictions
    outputs = model(x_valid_tensor)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)


    # # Calculate accuracy
    # accuracy = (predicted == y_valid).sum().item() / y_valid.size(0)
    # print("Validation Accuracy:", accuracy)
    
    # # Compute confusion matrix
    # cm = confusion_matrix(y_valid.numpy(), predicted.numpy())
    # print("Confusion Matrix:\n", cm)

    # # Calculate overall metrics
    # TP = cm.diagonal().sum()  # Total True Positives
    # FP = cm.sum(axis=0).sum() - TP  # Total False Positives
    # FN = cm.sum(axis=1).sum() - TP  # Total False Negatives
    # TN = cm.sum() - (TP + FP + FN)  # Total True Negatives

    # # Overall Precision, Recall, and F1 score
    # precision = TP / (TP + FP + 1e-10)  # Adding epsilon to avoid division by zero
    # recall = TP / (TP + FN + 1e-10)
    # f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    # # Print metrics
    # print(f"Overall Precision: {precision}")
    # print(f"Overall Recall: {recall}")
    # print(f"Overall F1 Score: {f1_score}")

    # # Save accuracy and metrics to a file
    # file_path = "validation_metrics_overall.txt"
    # with open(file_path, "w") as file:
    #     file.write(f"Validation Accuracy: {accuracy}\n")
    #     file.write("Confusion Matrix:\n")
    #     file.write("\n".join(["\t".join(map(str, row)) for row in cm]))
    #     file.write("\n\nMetrics:\n")
    #     file.write(f"Overall Precision: {precision}\n")
    #     file.write(f"Overall Recall: {recall}\n")
    #     file.write(f"Overall F1 Score: {f1_score}\n")
