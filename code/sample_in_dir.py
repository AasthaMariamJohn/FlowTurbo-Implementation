import os
import torch
import time
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models_assemble import FlowTurboAssemble

from PIL import Image
from IPython.display import display
torch.set_grad_enabled(False)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

# Configuration variables
cfg_scale = 4
class_labels = [207, 360, 387, 974, 88, 979, 417, 279]  # Example class labels
image_size = "256"
refiner_ckpt = 'SiT-XL-2-Refiner.pt'
predictor_ckpt = 'SiT-XL-2-256x256.pt'
vae_ckpt = "stabilityai/sd-vae-ft-ema"

latent_size = int(image_size) // 8

# Load VAE model
vae = AutoencoderKL.from_pretrained(vae_ckpt).to(device)

# Seed for reproducibility
seed = 0
torch.manual_seed(seed)
samples_per_row = 4

# Create sampling noise:
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance:
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sampling configuration
sample_config = [{'N_H': 1, 'N_P': 5, 'N_R': 3, 'SAC': False}]
method_str = ''.join([f"{key}{value}" for key, value in sample_config[0].items()])

# Initialize FlowTurbo model
FlowTurbo = FlowTurboAssemble(predictor_ckpt=predictor_ckpt, refiner_ckpt=refiner_ckpt, vae_ckpt=vae_ckpt, **sample_config[0])
FlowTurbo.eval()
FlowTurbo.to(device)

# Output directory
output_dir = "./samples"
os.makedirs(output_dir, exist_ok=True)

# Generate and save images
with torch.autocast(device_type="cuda"):
    imgs, samples = FlowTurbo(z, **model_kwargs)
    
    # Save each image separately in the samples folder
    for i, img in enumerate(imgs):
        image_path = os.path.join(output_dir, f'sample_{method_str}_{i}.png')
        save_image(img, image_path, nrow=samples_per_row, normalize=True, value_range=(-1, 1))
        print(f"Image {i} is saved in {image_path}")
