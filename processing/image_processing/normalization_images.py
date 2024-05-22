import os
import numpy as np
from PIL import Image
import torch

# Define normalization values
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Function to resize, center crop, normalize, and convert an image to a PyTorch tensor
def load_and_transform_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format

    # Resize the smaller edge to 256 while preserving aspect ratio
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int(height * (256 / width))
    else:
        new_height = 256
        new_width = int(width * (256 / height))
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Center crop to 224x224 pixels
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    image = image.crop((left, top, right, bottom))

    # Convert image to numpy array and normalize
    image_np = np.array(image) / 255.0  # Scale to [0, 1]
    image_np = (image_np - mean) / std  # Normalize

    # Convert numpy array to PyTorch tensor and permute dimensions to [C, H, W]
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).float()

    return image_tensor

# Function to save a tensor as an image
def save_tensor_as_image(tensor, save_path):
    # Denormalize the tensor
    tensor = tensor.permute(1, 2, 0).numpy()  # Change to [H, W, C] for PIL
    tensor = (tensor * std) + mean  # Denormalize
    tensor = np.clip(tensor, 0, 1)  # Clip values to [0, 1]
    tensor = (tensor * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    image = Image.fromarray(tensor)
    image.save(save_path)

# Example usage
image_dir = '../../../data/textbook_of_pathology_image/'  # Directory containing your images
processed_image_dir = '../../data/images/'  # Directory to save processed images

os.makedirs(processed_image_dir, exist_ok=True)  # Create directory if it doesn't exist

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    if image_path.lower().endswith(('png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif')):  # Check for valid image extensions
        normalized_image = load_and_transform_image(image_path)
        save_path = os.path.join(processed_image_dir, image_name)
        save_tensor_as_image(normalized_image, save_path)

print(f"Processed images saved in {processed_image_dir}")
