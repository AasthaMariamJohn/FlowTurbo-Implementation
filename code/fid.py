import torch
import numpy as np
from PIL import Image
from scipy import linalg
from torchvision import models, transforms

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load Inception model (without final FC layer)
inception = models.inception_v3(weights="Inception_V3_Weights.DEFAULT", transform_input=False).eval()
inception.fc = torch.nn.Identity()

# Function to calculate FID for two images
def calculate_fid(image1_path, image2_path):
    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    # Get features from the Inception model
    with torch.no_grad():
        features1 = inception(img1).cpu().numpy().flatten()
        features2 = inception(img2).cpu().numpy().flatten()

    # FID for single images is the squared Euclidean distance between features
    fid = np.sum((features1 - features2) ** 2)
    return fid

# Example usage
image1_path = '/content/FlowTurbo-Implementation/code/sample.png'
image2_path = '/content/FlowTurbo-Implementation/code/sample_N_H1N_P5N_R3SACFalse.png'
fid_score = calculate_fid(image1_path, image2_path)/60
print(f'FID score between the two images: {fid_score}')




