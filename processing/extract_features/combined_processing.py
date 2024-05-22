import os
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
import json
from transformers import BertTokenizer

# Load pre-trained ResNet model and remove the final fully connected layer
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# Define image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Filter data for only yes/no questions and map image IDs
    filtered_data = []
    image_ids = set()
    for item in data:
        answer = item['Answers'].strip().lower()
        if answer in ['yes', 'no']:
            filtered_data.append(item)
            image_ids.add(f"img_{item['Image_ID'].split('.')[-1]}")

    return filtered_data, image_ids


# Load and filter data
data_path = '../../../data/QA_pairs_vb.json'  # Update with your actual path
data, relevant_image_ids = load_data(data_path)

# Extract features from relevant images
image_folder_path = "../../data/images/"
features_list = []
for filename in os.listdir(image_folder_path):
    img_id = filename.split('.')[0]
    if img_id in relevant_image_ids:
        img_path = os.path.join(image_folder_path, filename)
        img = Image.open(img_path)
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            img_features = resnet(img_tensor)

        features_list.append(img_features.flatten().numpy())

# Convert the list of features to a numpy array
features_array = np.array(features_list)
np.save("../../data/features_images/image_features.npy", features_array)

# Tokenize questions
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
questions = [item['Questions'] for item in data]
answers = torch.tensor([1 if item['Answers'].strip().lower() == 'yes' else 0 for item in data])

tokenized_questions = tokenizer(questions, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Save tokenized questions and answers
torch.save(tokenized_questions, '../../data/text/tokenized_questions_yn.pt')
torch.save(answers, '../../data/text/tokenized_answers_yn.pt')
