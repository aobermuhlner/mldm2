import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load image features
image_features = np.load('image_features.npy')

# Load tokenized questions and answers
tokenized_questions = torch.load('tokenized_questions.pt')
tokenized_answers = torch.load('tokenized_answers.pt')

# Convert image features to torch tensor
image_features = torch.tensor(image_features, dtype=torch.float32)

# Load BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Assuming tokenized_questions['input_ids'] has the shape [num_samples, seq_length]
# and image_features has the shape [num_samples, img_feature_dim]

# Get the BERT embeddings for the questions (mean of token embeddings)
with torch.no_grad():
    question_embeddings = bert_model(**tokenized_questions).last_hidden_state.mean(dim=1)

# Concatenate image features and question embeddings
combined_features = torch.cat((image_features, question_embeddings), dim=1)

class CombinedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(CombinedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Define the input dimension as the sum of image feature dimension and BERT embedding dimension
input_dim = image_features.shape[1] + question_embeddings.shape[1]

# Initialize the model, loss function, and optimizer
model = CombinedClassifier(input_dim=input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create a dataset and dataloader
dataset = TensorDataset(combined_features, tokenized_answers)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_features, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
