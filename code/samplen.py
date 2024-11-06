import torch
import time
import argparse
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models_assemble import FlowTurboAssemble

from PIL import Image
from IPython.display import display
torch.set_grad_enabled(False)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("GPU not found. Using CPU instead.")

    # Load the VAE model
    vae = AutoencoderKL.from_pretrained(args.vae_ckpt).to(device)

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Prepare class labels and latent noise
    latent_size = args.image_resolution // 8
    z = torch.randn(args.num_samples, 4, latent_size, latent_size, device=device)
    y = torch.tensor(args.class_labels, device=device)

    # Setup classifier-free guidance
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * args.num_samples, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Initialize FlowTurbo model with sampling configuration
    sample_config = [{'N_H': 1, 'N_P': 5, 'N_R': 3, 'SAC': args.adaptive_sampling}]
    method_str = ''.join([f"{key}{value}" for key, value in sample_config[0].items()])

    FlowTurbo = FlowTurboAssemble(
        predictor_ckpt=args.predictor_ckpt, 
        refiner_ckpt=args.refiner_ckpt, 
        vae_ckpt=args.vae_ckpt, 
        **sample_config[0]
    )
    FlowTurbo.eval()
    FlowTurbo.to(device)

    # Initialize adaptive step size parameters
    C = 0.1  # Adjust constant for scaling based on experimentation
    adaptive_failed = False  # Flag to track if adaptive sampling fails
    fixed_step = 0.05  # Define fixed step size for fallback

    # Generate and save images with optional adaptive sampling
    with torch.autocast(device_type="cuda"):
        for t in range(args.num_timesteps):
            if not adaptive_failed and args.adaptive_sampling:
                try:
                    # Estimate gradient or variability
                    grad_magnitude = torch.norm(FlowTurbo.compute_gradients(z, t), dim=(1, 2, 3))
                    
                    # Calculate adaptive step size
                    adaptive_step = torch.clamp(C / (grad_magnitude + 1e-5), min=1e-3, max=0.1)
                    
                    # Adjust sampling parameters based on adaptive step
                    for i, step in enumerate(adaptive_step):
                        if step < 0.05:
                            FlowTurbo.N_R = max(1, FlowTurbo.N_R - 1)  # Reduce refinement if variability is low
                        else:
                            FlowTurbo.N_R = min(FlowTurbo.N_R + 1, max_steps)  # Increase if variability is high
                    
                    # Perform adaptive sampling step
                    z = FlowTurbo(z, **model_kwargs)
                
                except Exception as e:
                    print(f"Adaptive sampling failed at timestep {t} due to error: {e}")
                    print("Reverting to fixed-step sampling for the remaining steps.")
                    adaptive_failed = True  # Trigger fallback to fixed-step sampling
            else:
                # Fallback: Use fixed-step sampling if adaptive sampling fails
                z = FlowTurbo(z + FlowTurbo.step(z, t, fixed_step), **model_kwargs)

        # Save generated images
        image_path = f"{args.output_dir}/sample_{method_str}.png"
        save_image(z, image_path, nrow=int(args.samples_per_row), normalize=True, value_range=(-1, 1))
        print(f"Images saved to {image_path}")

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
