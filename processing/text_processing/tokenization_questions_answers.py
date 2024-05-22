import json
from transformers import BertTokenizer
import torch

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def tokenize_qa(data, tokenizer, max_length=128):
    questions = [item['Questions'] for item in data]
    answers = [1 if item['Answers'].strip().lower() == 'yes' else 0 for item in data if item['Answers'].strip().lower() in ['yes', 'no']]

    # Tokenize questions
    tokenized_questions = tokenizer(questions, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    return tokenized_questions, torch.tensor(answers)

# Example usage
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data_path = '../../../data/QA_pairs_vb.json'  # Update with your actual path
data = load_data(data_path)
tokenized_questions, tokenized_answers = tokenize_qa(data, tokenizer)

# Print tokenized questions and answers for verification
print(tokenized_questions)
print(tokenized_answers)

# Save the tokenized questions and answers
torch.save(tokenized_questions, '../../data/text/tokenized_questions.pt')
torch.save(tokenized_answers, '../../data/text/tokenized_answers.pt')
