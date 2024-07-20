import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load and preprocess data
with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(len(X_train) * split_ratio)
X_test = X_train[split_index:]
y_test = y_train[split_index:]
X_train = X_train[:split_index]
y_train = y_train[:split_index]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

train_dataset = ChatDataset(X_train, y_train)
test_dataset = ChatDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store loss values
loss_values = []

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    loss_values.append(loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

# Plotting the loss values
plt.plot(range(num_epochs), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

# Example data (You would replace these with your actual data)
epochs = np.arange(1, 1001)  # 1000 epochs
loss = np.random.uniform(0.01, 0.1, 1000) * np.exp(-epochs / 150)  # Simulated loss decreasing exponentially

# Assuming you might have accuracy data
accuracy = 1 - (np.random.uniform(0, 0.02, 1000) * np.exp(-epochs / 200))  # Simulated accuracy improving

# Create a figure and a set of subplots
fig, ax1 = plt.subplots()

# Plotting training loss
color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs, loss, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Instantiate a second y-axis for the same x-axis
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  # We already handled the x-label with ax1
ax2.plot(epochs, accuracy, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Add a title and show the plot
plt.title('Training Loss and Accuracy Over Epochs')
fig.tight_layout()  # To ensure there's no overlap in labels
plt.show()




data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')

# Evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for words, labels in data_loader:
            words = words.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(words)
            _, predicted = torch.max(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return predictions, true_labels

# Reload the model for evaluation
model.load_state_dict(torch.load(FILE)["model_state"])

# Evaluate the model and calculate F1 score
predictions, true_labels = evaluate_model(model, test_loader)
f1 = f1_score(true_labels, predictions, average='weighted')
print(f'F1 Score: {f1:.4f}')

