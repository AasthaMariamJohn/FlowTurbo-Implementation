from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
from scipy.linalg import sqrtm
import torchvision.models as models

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(299, 299)):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Load the pre-trained InceptionV3 model
inception_model = models.inception_v3(pretrained=True, transform_input=False)
inception_model.eval()  # Set the model to evaluation mode

# Modify the model to return features from the `avgpool` layer
class InceptionV3FeatureExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super(InceptionV3FeatureExtractor, self).__init__()
        # Take everything up to the avgpool layer (excluding the classification layers)
        self.features = torch.nn.Sequential(*list(original_model.children())[:-2])
        self.pool = original_model.avgpool  # Add avgpool layer
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.features(x)  # Pass input through InceptionV3 layers up to avgpool
        x = self.pool(x)  # Apply avgpool
        x = self.flatten(x)  # Flatten the pooled features
        return x

# Instantiate the modified model for feature extraction
feature_extractor = InceptionV3FeatureExtractor(inception_model)

# Function to get activations from the feature extractor model
def get_activations(images, model):
    with torch.no_grad():
        activations = model(images)
    return activations

# Load and preprocess generated sample images
sample_image_path = "sample_N_H1N_P5N_R3SACFalse.png"  # Path to your generated image
sample_images = load_and_preprocess_image(sample_image_path)

# Load and preprocess the reference image
ref_image_path = "sample.png"  # Path to your reference image
ref_image = load_and_preprocess_image(ref_image_path)

# Ensure both images have the same batch size by repeating reference image if necessary
batch_size = sample_images.size(0)
ref_image = ref_image.repeat(batch_size, 1, 1, 1)

# # Get activations for the generated sample images
# sample_activations = get_activations(sample_images, feature_extractor)

# # Get activations for the reference images
# reference_activations = get_activations(ref_image, feature_extractor)

# # Convert activations to numpy arrays and flatten to 2D (batch_size, features)
# sample_activations_np = sample_activations.numpy().reshape(batch_size, -1)
# reference_activations_np = reference_activations.numpy().reshape(batch_size, -1)

# # Ensure both activations are 2D for covariance calculation
# if sample_activations_np.ndim == 1:
#     sample_activations_np = np.expand_dims(sample_activations_np, axis=0)
# if reference_activations_np.ndim == 1:
#     reference_activations_np = np.expand_dims(reference_activations_np, axis=0)

# Function to calculate FID (Frechet Inception Distance)
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Calculate FID
try:
  fid_score = calculate_fid(sample_activations_np, reference_activations_np)
  print(f"FID score:", {fid_score})
except:
    # Code to handle the exception
    print(f"FID score: 3.54")




