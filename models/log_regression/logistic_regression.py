import torch
import numpy as np
from transformers import BertModel, BertTokenizer

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

# Get the BERT embeddings for the questions (mean of token embeddings)
with torch.no_grad():
    question_embeddings = bert_model(**tokenized_questions).last_hidden_state.mean(dim=1)


yes_no_indices = [i for i, ans in enumerate(tokenized_answers) if ans in [0, 1]]
filtered_image_features = image_features[yes_no_indices]
filtered_question_embeddings = question_embeddings[yes_no_indices]
filtered_answers = tokenized_answers[yes_no_indices]


# Concatenate image features and question embeddings
combined_features = torch.cat((filtered_image_features, filtered_question_embeddings), dim=1).numpy()
filtered_answers = filtered_answers.numpy()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, filtered_answers, test_size=0.2, random_state=42)

# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Train the model
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

