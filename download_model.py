# download_model.py
import torch
import torchvision.models as models
import os

print("Downloading DenseNet-121 model...")
# Download the model weights
model = models.densenet121(pretrained=True)

# Define the path to save the model
model_dir = "/app/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "densenet121-pretrained.pth")

# Save the model's state dictionary
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")