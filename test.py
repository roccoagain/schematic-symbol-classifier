# use this script to verify the model correctly identifies the symbols

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import random

# Define the same transformations used during training
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),  # Resize images
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels if needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize
    ]
)

# Load class names (make sure they are in the same order as training)
classes = [
    "Ammeter",
    "ac_src",
    "battery",
    "cap",
    "curr_src",
    "dc_volt_src_1",
    "dc_volt_src_2",
    "dep_curr_src",
    "dep_volt",
    "diode",
    "gnd_1",
    "gnd_2",
    "inductor",
    "resistor",
    "voltmeter",
]  # Update if necessary


# Load model structure
class SymbolClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SymbolClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Load the trained model
num_classes = len(classes)
model = SymbolClassifier(num_classes)
model.load_state_dict(
    torch.load("symbol_classifier.pth", map_location=torch.device("cpu"), weights_only=True)
)
model.eval()  # Set model to evaluation mode

# Select a random image from the dataset
data_dir = "SolvaDataset_200_v3"
category = random.choice(os.listdir(data_dir))  # Pick a random class folder
image_path = os.path.join(
    data_dir, category, random.choice(os.listdir(os.path.join(data_dir, category)))
)

if not os.path.exists(image_path):
    print(f"Error: Image '{image_path}' not found!")
else:
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    # Print result
    print(f"Randomly selected image: {image_path}")
    print(f"Predicted class: {classes[predicted_class]}")
