import torchvision.models as models
import torch

# Load pre-trained ResNet18 model
resnet = models.resnet18(pretrained=True)

# Print the original ResNet model architecture
print(resnet)

# Output will show that the last layer is a fully connected layer:
# (fc): Linear(in_features=512, out_features=1000, bias=True)

# Remove the final fully connected layer
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# Print the modified ResNet model architecture
print(resnet)

# Output will show that the final fully connected layer has been removed:
# Sequential(
#   (0): Conv2d(...)
#   (1): BatchNorm2d(...)
#   (2): ReLU(...)
#   (3): MaxPool2d(...)
#   ...
#   (9): AdaptiveAvgPool2d(output_size=(1, 1))
# )
