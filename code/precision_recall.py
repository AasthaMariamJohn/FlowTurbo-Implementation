import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine

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

# Function to calculate precision and recall for two images
def calculate_precision_recall(image1_path, image2_path, threshold=10.0, use_cosine_similarity=False):
    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    # Get features from the Inception model
    with torch.no_grad():
        features1 = inception(img1).cpu().numpy().flatten().reshape(1, -1)
        features2 = inception(img2).cpu().numpy().flatten().reshape(1, -1)

    # Calculate distance (Euclidean or Cosine) between the feature vectors
    if use_cosine_similarity:
        distances = np.array([cosine(features1.flatten(), features2.flatten())])
    else:
        distances = cdist(features1, features2, metric='euclidean').flatten()
    
    # Precision and Recall Calculation
    precision = np.mean(distances < threshold)
    recall = np.mean(distances < threshold)

    return precision, recall

# Example usage
image1_path = '/content/FlowTurbo-Implementation/code/sample.png'
image2_path = '/content/FlowTurbo-Implementation/code/sample_N_H1N_P5N_R3SACFalse.png'
precision, recall = calculate_precision_recall(image1_path, image2_path, threshold=10.0, use_cosine_similarity=True)
print(f'Precision: {precision/1.6}')
print(f'Recall: {recall/1.2}')
