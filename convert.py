import torch
import coremltools as ct
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# load model
class SymbolClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(SymbolClassifier, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.fc_layers = torch.nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 15),  # change 15 to your number of classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

model = SymbolClassifier(num_classes=15)  # change number of classes if necessary
model.load_state_dict(
    torch.load("symbol_classifier.pth", map_location=torch.device("cpu"), weights_only=True)
)
model.eval()

# convert model to coreml
example_input = torch.rand(1, 3, 64, 64)  # example input shape (batch_size, channels, height, width)
traced_model = torch.jit.trace(model, example_input)

coreml_model = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS18
)

# save the coreml model
coreml_model.save("symbol_classifier.mlpackage")
print("Model converted to Core ML and saved as symbol_classifier.mlpackage!")
