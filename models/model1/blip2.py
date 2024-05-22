import os
import json
import torch
import pandas as pd
from PIL import Image
from transformers import BlipProcessor
from datasets import Dataset, DatasetDict

# Set up paths
image_features_path = '../../data/unprocessed_images/'
tokenized_answers_path = "../../data/text/questions_answers.json"

# Load questions and answers
with open(tokenized_answers_path, 'r') as f:
    qa_data = json.load(f)

# Function to map Image_ID to image file name
def get_image_file_name(image_id):
    try:
        # Extract the number after 'img_' or 'Fig.'
        if image_id.startswith('img_') or image_id.startswith('Fig.'):
            image_number = image_id.split('_')[1] if image_id.startswith('img_') else image_id.split('.')[1]  # 'img_16' -> '16' or 'Fig.298' -> '298'
            return f"img_{int(image_number)}.jpg"
        else:
            print(f"Unexpected format for image_id: {image_id}")
            return None
    except IndexError:
        print(f"Unexpected format for image_id: {image_id}")
        return None
    except ValueError:
        print(f"Cannot convert to integer: {image_id.split('_')[1]}")
        return None


# Prepare dataset entries
dataset_entries = []

for item in qa_data:
    image_id = item['Image_ID']
    question = item['Questions']
    answer = item['Answers']

    # Get the corresponding image file name
    image_file_name = get_image_file_name(image_id)
    if image_file_name is None:
        continue

    image_path = os.path.join(image_features_path, image_file_name)

    # Load the image
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping...")
        continue

    dataset_entries.append({
        'image_path': image_path,
        'question': question,
        'answer': answer
    })

# Check if dataset_entries is populated correctly
if len(dataset_entries) == 0:
    print("No valid entries found. Please check your data and paths.")

# Convert to Dataset
dataset = Dataset.from_pandas(pd.DataFrame(dataset_entries))

# Split into train and test sets
dataset = dataset.train_test_split(test_size=0.1)
