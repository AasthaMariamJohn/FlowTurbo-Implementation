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
image_size = 256
refiner_ckpt = 'SiT-XL-2-Refiner.pt'
predictor_ckpt = 'SiT-XL-2-256x256.pt'
vae_ckpt = "stabilityai/sd-vae-ft-ema"

latent_size = image_size // 8

# Load VAE model
vae = AutoencoderKL.from_pretrained(vae_ckpt).to(device)

# Seed for reproducibility
seed = 0
torch.manual_seed(seed)
samples_per_row = 4

# Create sampling noise
n = len(class_labels)
z = torch.randn(n, 4, latent_size, latent_size, device=device)
y = torch.tensor(class_labels, device=device)

# Setup classifier-free guidance
z = torch.cat([z, z], 0)
y_null = torch.tensor([1000] * n, device=device)
y = torch.cat([y, y_null], 0)
model_kwargs = dict(y=y, cfg_scale=cfg_scale)

# Sampling configuration
sample_config = [{'N_H': 1, 'N_P': 5, 'N_R': 3, 'SAC': True}]
method_str = ''.join([f"{key}{value}" for key, value in sample_config[0].items()])

# Initialize FlowTurbo model
FlowTurbo = FlowTurboAssemble(
    predictor_ckpt=predictor_ckpt,
    refiner_ckpt=refiner_ckpt,
    vae_ckpt=vae_ckpt,
    **sample_config[0]
)
FlowTurbo.eval()
FlowTurbo.to(device)

# Adaptive sampling parameters
C = 0.1  # Constant for adaptive step calculation
adaptive_failed = False  # Flag for fallback
fixed_step = 0.05  # Fixed step size fallback

# Output directory
output_dir = "./samples"
os.makedirs(output_dir, exist_ok=True)

# Generate and save images
with torch.autocast(device_type="cuda"):
    try:
        # Adaptive sampling process
        for t in range(50):  # Assuming 50 timesteps
            if sample_config[0]['SAC'] and not adaptive_failed:
                try:
                    # Estimate gradient or variability for adaptive step calculation
                    grad_magnitude = torch.norm(FlowTurbo.compute_gradients(z, t), dim=(1, 2, 3))
                    adaptive_step = torch.clamp(C / (grad_magnitude + 1e-5), min=1e-3, max=0.1)

                    # Adjust number of refinement steps based on gradient
                    for i, step in enumerate(adaptive_step):
                        FlowTurbo.N_R = max(1, FlowTurbo.N_R - 1) if step < 0.05 else min(FlowTurbo.N_R + 1, 10)
                    
                    # Perform sampling step with adaptive adjustments
                    z = FlowTurbo(z, **model_kwargs)

                except Exception as e:
                    print(f"Adaptive sampling failed at timestep {t} due to error: {e}")
                    print("Reverting to fixed-step sampling.")
                    adaptive_failed = True  # Switch to fixed-step sampling on failure

            else:
                # Fallback: fixed-step sampling if adaptive fails
                z = FlowTurbo(z + FlowTurbo.step(z, t, fixed_step), **model_kwargs)

        # Save each generated image
        for i, img in enumerate(z):
            image_path = os.path.join(output_dir, f'sample_{method_str}_{i}.png')
            save_image(img, image_path, nrow=samples_per_row, normalize=True, value_range=(-1, 1))
            print(f"Image {i} is saved in {image_path}")

    except Exception as e:
        print(f"Sampling failed due to error: {e}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-sampling", action="store_true", help="Enable adaptive sampling.")
    parser.add_argument("--predictor_ckpt", type=str, required=True, help="Path to the predictor checkpoint.")
    parser.add_argument("--refiner_ckpt", type=str, required=True, help="Path to the refiner checkpoint.")
    parser.add_argument("--vae_ckpt", type=str, default="stabilityai/sd-vae-ft-ema", help="Path to the VAE checkpoint.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of images to generate.")
    parser.add_argument("--output_dir", type=str, default="./samples", help="Directory to save generated images.")
    parser.add_argument("--image_resolution", type=int, default=256, help="Resolution of the generated images.")
    parser.add_argument("--cfg_scale", type=float, default=4, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--class_labels", type=int, nargs='+', default=[207, 360, 387, 974, 88, 979, 417, 279], help="List of class labels.")
    parser.add_argument("--samples_per_row", type=int, default=4, help="Number of samples per row in saved image grid.")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of timesteps in the sampling process.")
    
    args = parser.parse_args()
    main(args)
