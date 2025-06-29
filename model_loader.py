"""
jitloader/model_loader.py

Entry point for JIT model loading pipeline.
Loads only metadata from .safetensors using safetensor_metadata,
creates dummy blueprints, and initializes InferenceScheduler for each model.
TODO: Replace DummyModel with actual model blueprints (CLIP, T5, VAE, Flux).
"""
import json
from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel
import comfy.ops as ops

# Assuming these paths are valid within the ComfyUI environment
from comfy.ldm.models.autoencoder import AutoencoderKL
from comfy.ldm.flux.model import Flux

from .safetensor_metadata import extract_safetensor_metadata
from .scheduler import FluxScheduler, VAEScheduler, T5Scheduler, CLIPScheduler
from .mem_allocator import MemoryAllocator


def _load_vae_model_config():
    """
    Returns the hardcoded VAE configuration.
    """
    return {
        "embed_dim": 16,
        "ddconfig": {
            "double_z": True, "z_channels": 16, "resolution": 1024, "in_channels": 3,
            "out_ch": 3, "ch": 128, "ch_mult": [1, 2, 4, 4], "num_res_blocks": 2,
            "attn_resolutions": [], "dropout": 0.0
        }
    }


def _load_flux_model_config():
    """
    Returns the hardcoded FLUX model configuration.
    """
    return {
        "in_channels": 16, "out_channels": 16, "hidden_size": 2048,
        "depth": 28, "num_heads": 32, "patch_size": 2,
        "context_in_dim": 4096, "vec_in_dim": 512, "mlp_ratio": 4.0,
        "qkv_bias": True, "depth_single_blocks": 28, "axes_dim": [64],
        "theta": 10000, "guidance_embed": True,
    }

def _load_t5_config(path: str):
    """
    Correctly loads a model's config.json file.
    """
    with open(path, 'r') as f:
        return json.load(f)

def _load_clip_config(path: str):
    """
    Infers the CLIPTextConfig parameters by inspecting tensor shapes
    in the safetensor header.
    """
    header = extract_safetensor_metadata(path)
    
    # From text_model.embeddings.token_embedding.weight
    embedding_info = header['text_model.embeddings.token_embedding.weight']
    vocab_size, hidden_size = embedding_info['shape']

    # From text_model.encoder.layers.0.mlp.fc1.weight
    mlp_info = header['text_model.encoder.layers.0.mlp.fc1.weight']
    intermediate_size = mlp_info['shape'][0]

    # Find the highest layer index to determine the number of layers
    layer_keys = [k for k in header if k.startswith('text_model.encoder.layers.')]
    layer_indices = {int(k.split('.')[3]) for k in layer_keys}
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 0

    # We assume some standard values like num_attention_heads if they can't be inferred
    return {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 12, # Standard for CLIP-L
        "projection_dim": 768    # Standard for CLIPTextModel
    }



def _load_t5_weight_map(path: str):
    """
    Loads the T5 weight map from its specific JSON index file.
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_pipeline(device: str = "cuda", quant_config: str = None, cpu_pool_size: int = 2*1024*1024*1024, gpu_pool_size: int = 4*1024*1024*1024):
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

    # --- Create Model Blueprints on 'meta' device ---
    # This avoids allocating memory. The scheduler will stream weights later.
    clip_config = CLIPTextConfig(**_load_clip_config(clip_path))

    clip_blueprint = CLIPTextModel(clip_config).to("meta")

    t5_config_dict = _load_t5_config(t5_path)
    t5_config = T5Config.from_dict(t5_config_dict)
    t5_config.use_cache = False
    t5_blueprint = T5EncoderModel(t5_config).to("meta")


    vae_config = _load_vae_model_config()
    vae_blueprint = AutoencoderKL(**vae_config).to("meta")

    flux_config = _load_flux_model_config()
    flux_blueprint = Flux(**flux_config, operations=ops.disable_weight_init).to("meta")

    # Initialize memory allocator
    allocator = MemoryAllocator(cpu_pool_size, gpu_pool_size, device)

    # Initialize schedulers
    clip_scheduler = CLIPScheduler(clip_path, clip_blueprint, allocator, device=device, quant_config=quant_config)
    t5_model_dir = "jitloader/t5"
    t5_weight_map = _load_t5_weight_map("jitloader/t5/model.safetensors.index.json")
    t5_scheduler = T5Scheduler(t5_model_dir, t5_blueprint, allocator, device=device, model_config=t5_weight_map, quant_config=quant_config)
    vae_scheduler = VAEScheduler(vae_path, vae_blueprint, allocator, device=device, quant_config=quant_config)
    flux_scheduler = FluxScheduler(flux_path, flux_blueprint, allocator, device=device, quant_config=quant_config)

    return {
        "clip": clip_scheduler,
        "t5": t5_scheduler,
        "vae": vae_scheduler,
        "flux": flux_scheduler,
    }


if __name__ == "__main__":
    schedulers = load_pipeline()
    print("Schedulers initialized:", list(schedulers.keys()))