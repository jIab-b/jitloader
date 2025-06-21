"""
jitloader/inference.py

End-to-end inference script for the JIT pipeline.
- Loads the pre-configured schedulers.
- Tokenizes a sample prompt.
- Generates a latent noise tensor.
- Executes the T5, CLIP, FLUX, and VAE models in sequence.
- Saves the final image to disk.
"""
import torch
import numpy as np
from PIL import Image
from transformers import T5Tokenizer, CLIPTokenizer

from .model_loader import load_pipeline

def run_inference(prompt: str, output_path: str = "jitloader/output.png"):
    """
    Runs the full JIT inference pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load all model schedulers
    schedulers = load_pipeline(device)
    flux_scheduler = schedulers["flux"]
    vae_scheduler = schedulers["vae"]
    # TODO: Implement CLIP and T5 scheduler execution for text encoding.

    # 2. Prepare inputs
    # TODO: Replace with actual tokenizer logic
    # For now, create dummy context tensors.
    # This is a placeholder for the concatenated output of T5 and CLIP.
    context = torch.randn(1, 1024, 4096, device=device) 
    # This is a placeholder for the pooled vector embeddings.
    y = torch.randn(1, 2048, device=device)

    # Latent noise
    height, width = 1024, 1024
    patch_size = flux_scheduler.blueprint.patch_size
    latent_height = height // patch_size
    latent_width = width // patch_size
    latent_channels = flux_scheduler.blueprint.in_channels
    latents = torch.randn(1, latent_channels, latent_height, latent_width, device=device)

    # Timestep
    timestep = torch.tensor([1000], device=device)

    # 3. Run the FLUX model
    print("Running FLUX model...")
    output_latents = flux_scheduler.run_inference(latents, timestep, context, y)
    print("FLUX model finished.")

    # 4. Decode the latents with the VAE
    # The VAE expects a different channel dimension.
    # This is a simplified call; a real implementation would need to match
    # the expected input shape and call the decoder layers sequentially.
    # For now, we assume a direct decode method for simplicity.
    # output_latents needs to be scaled correctly before decoding.
    # VAE blueprint: AutoencoderKL
    # We need to call the 'decode' method layer by layer.
    # This part is complex and is stubbed for now.
    print("Running VAE decoder (STUBBED)...")
    # vae_scheduler.decode(output_latents) # This is a placeholder
    
    # As a placeholder, create a dummy image from the output latents
    # This will look like noise, but proves the pipeline ran.
    # We'll take the first 3 channels to create an RGB image.
    noisy_image_tensor = output_latents[0, :3, :, :].cpu()
    noisy_image_tensor = (noisy_image_tensor + 1.0) / 2.0 # Normalize to [0, 1]
    noisy_image_tensor = noisy_image_tensor.permute(1, 2, 0).numpy()
    noisy_image_tensor = (noisy_image_tensor * 255).astype(np.uint8)
    image = Image.fromarray(noisy_image_tensor)
    
    # 5. Save the output image
    image.save(output_path)
    print(f"Inference complete. Noisy output saved to {output_path}")


if __name__ == "__main__":
    sample_prompt = "A photograph of a majestic lion in the savanna."
    run_inference(sample_prompt)