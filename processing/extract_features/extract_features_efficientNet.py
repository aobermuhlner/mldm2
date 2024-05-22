import os
from torchvision import transforms
from PIL import Image
import torch
import torchvision.models as models
import numpy as np

# Load a pre-trained EfficientNet model (EfficientNet-B0 in this example)
efficientnet = models.efficientnet_b0(pretrained=True)

# Remove the final fully connected layer
efficientnet.classifier = torch.nn.Identity()

# Define image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Path to the folder containing images
image_folder_path = "../../data/images/"

# List to store extracted features
features_list = []

# Loop through all images in the folder
for filename in os.listdir(image_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Ensure to process only image files
        img_path = os.path.join(image_folder_path, filename)

        # Load and preprocess the image
        img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB mode
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Extract features
        with torch.no_grad():
            img_features = efficientnet(img_tensor)

        # Convert the extracted features to a list or a different format if needed
        features_list.append(img_features.flatten().numpy())

# Convert the list of features to a numpy array for further processing if needed
features_array = np.array(features_list)

# Save features to a file if needed
np.save("../../data/features_images/image_features2.npy", features_array)

print("Feature extraction complete. Features saved to ../../data/features_images/image_features2.npy")
