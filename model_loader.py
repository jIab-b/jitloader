"""
jitloader/model_loader.py

Entry point for JIT model loading pipeline.
Loads only metadata from .safetensors using safetensor_metadata,
creates dummy blueprints, and initializes InferenceScheduler for each model.
TODO: Replace DummyModel with actual model blueprints (CLIP, T5, VAE, Flux).
"""
from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
import comfy.ops as ops

# Assuming these paths are valid within the ComfyUI environment
from comfy.ldm.models.autoencoder import AutoencoderKL
from comfy.ldm.flux.model import Flux

from .safetensor_metadata import extract_safetensor_metadata
from .scheduler import InferenceScheduler

def load_pipeline(device: str = "cuda"):
    """
    Loads metadata-only safetensor files and returns initialized schedulers.
    Returns:
        dict: mapping model names to InferenceScheduler instances.
    """
    # Dummy safetensor file paths
    clip_path = "jitloader/clip/model.safetensors"
    t5_path = "jitloader/t5/config.json"
    vae_path = "jitloader/vae/ae.safetensors"
    flux_path = "jitloader/transformer/flux1-dev.safetensors"

    # Extract metadata only
    clip_meta = extract_safetensor_metadata(clip_path)
    t5_meta = extract_safetensor_metadata(t5_path)
    vae_meta = extract_safetensor_metadata(vae_path)
    flux_meta = extract_safetensor_metadata(flux_path)

    # --- Create Model Blueprints on 'meta' device ---
    # This avoids allocating memory. The scheduler will stream weights later.
    # TODO: Replace dummy configs with actual parameters loaded from model metadata/config.json

    # Config for CLIP ViT-bigG
    clip_config = CLIPTextConfig(
        vocab_size=49408, hidden_size=1280, intermediate_size=5120,
        num_hidden_layers=32, num_attention_heads=20
    )
    clip_blueprint = CLIPTextModel(clip_config).to("meta")

    # Config for T5-XXL
    t5_config = T5Config(
        vocab_size=32128, d_model=4096, d_kv=64, d_ff=10240,
        num_layers=24, num_heads=64
    )
    t5_blueprint = T5EncoderModel(t5_config).to("meta")

    # Config for Flux VAE
    vae_config = {
        "embed_dim": 16,
        "ddconfig": {
            "double_z": True, "z_channels": 16, "resolution": 1024, "in_channels": 3,
            "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2,
            "attn_resolutions": [], "dropout": 0.0
        }
    }
    vae_blueprint = AutoencoderKL(**vae_config).to("meta")

    # Config for FLUX model
    flux_config = {
        "in_channels": 16,
        "out_channels": 16,
        "hidden_size": 2048,
        "depth": 28,
        "num_heads": 32,
        "patch_size": 2,
        "context_in_dim": 4096,
        "vec_in_dim": 512,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "depth_single_blocks": 28,
        "axes_dim": [64],
        "theta": 10000,
        "guidance_embed": True,
    }
    flux_blueprint = Flux(**flux_config, operations=ops.disable_weight_init).to("meta")

    # Initialize schedulers
    clip_scheduler = InferenceScheduler(clip_path, clip_blueprint, device=device)
    t5_scheduler = InferenceScheduler(t5_path, t5_blueprint, device=device)
    vae_scheduler = InferenceScheduler(vae_path, vae_blueprint, device=device)
    flux_scheduler = InferenceScheduler(flux_path, flux_blueprint, device=device)

    return {
        "clip": clip_scheduler,
        "t5": t5_scheduler,
        "vae": vae_scheduler,
        "flux": flux_scheduler,
    }


if __name__ == "__main__":
    schedulers = load_pipeline()
    print("Schedulers initialized:", list(schedulers.keys()))