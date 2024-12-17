# from kagglehub import kagglehub
# # Download latest version
# path = kagglehub.dataset_download("kazanova/sentiment140")

# print("Path to dataset files:", path)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
nltk.download('punkt')
nltk.download('punkt_tab')

def tokenizer(s):
    return word_tokenize(s)

def sigmoid(x):
  print(x)
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)


def train(epochs, learning_rate, batch_size, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    # Training loop
    for epoch in range(epochs):
        for i in tqdm(range(0, X_train.shape[0], batch_size)):
            try:
              print(i)
              X_batch = X_train[i:i + batch_size]
              y_batch = y_train[i:i + batch_size]
              print("Batch initialized")
              # Forward propagation
              hidden_layer_input = np.dot(X_batch, weights_input_hidden) + bias_hidden
              print("Hidden layer input 1")
              hidden_layer_output = sigmoid(hidden_layer_input)
              print("Hidden layer outptu 1")
              output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
              print("Output layer input 2")
              output_layer_output = sigmoid(output_layer_input)
              print("Fowrward propagation complete")
              # Calculate error
              error = y_batch - output_layer_output
              print("Start back propaagation")
              # Backpropagation
              d_output = error * sigmoid_derivative(output_layer_output)
              error_hidden_layer = d_output.dot(weights_hidden_output.T)
              d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
              print("Update weights and biases")
              # Update weights and biases
              weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
              bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
              weights_input_hidden += X_batch.T.dot(d_hidden_layer) * learning_rate
              bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
            except Exception as e:
              print(e)
        print(f"Epoch {epoch + 1}/{epochs}")

def val(epochs, learning_rate, batch_size, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_layer_input_val = np.dot(X_valid, weights_input_hidden) + bias_hidden
    hidden_layer_output_val = sigmoid(hidden_layer_input_val)
    output_layer_input_val = np.dot(hidden_layer_output_val, weights_hidden_output) + bias_output
    output_layer_output_val = sigmoid(output_layer_input_val)

    # Convert probabilities to predicted class labels
    y_pred_val = np.argmax(output_layer_output_val, axis=1)

    # Calculate accuracy
    accuracy = np.mean(y_pred_val == y_valid)
    print("Validation Accuracy:", accuracy)


df = pd.read_csv('/scratch/snf4za/social_context_nlp/sentiment140/versions/2/training.1600000.processed.noemoticon.csv',
                 encoding='latin-1') # Specify the encoding to 'latin-1'

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Split the temporary set into test and validation sets
test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Print the shapes of the resulting sets
print("Train set shape:", train_df.shape)
print("Test set shape:", test_df.shape)
print("Validation set shape:", val_df.shape)

y_train = train_df.iloc[:, 0]  # Selecting the first column as the label
X_train = train_df.iloc[:, 5]  # Selecting all columns except the first one as features
y_valid = val_df.iloc[:, 0]  # Selecting the first column as the label
X_valid = val_df.iloc[:, 5]  # Selecting all columns except the first one as features
y_test = test_df.iloc[:, 0]  # Selecting the first column as the label
X_test = test_df.iloc[:, 5]  # Selecting all columns except the first one as features


print("Start fitting data")
# =========================================
cv_train = CountVectorizer(lowercase=True,tokenizer=tokenizer)
train_vec_data = cv_train.fit_transform(X_train)
valid_vec_data = cv_train.transform(X_valid)
test_vec_data = cv_train.transform(X_test)
print("transformed")
X_train = train_vec_data
X_valid = valid_vec_data
X_test = test_vec_data

input_size = X_train.shape[1]
hidden_size = 2
output_size = len(np.unique(y_train))

np.random.seed(42)  # For reproducibility
weights_input_hidden = np.random.randn(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.randn(hidden_size, output_size)
bias_output = np.zeros((1, output_size))

print("Initialized")

# Training parameters
epochs = 1  # Number of training iterations
learning_rate = 0.1
batch_size = 100


print("Train data")
train(epochs, learning_rate, batch_size, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output).to("cuda")


# validation

print("Validation")
val(epochs, learning_rate, batch_size, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output).to("cuda")
# Now you have trained weights and biases for your neural network.
# You can use this to make predictions on new data.
