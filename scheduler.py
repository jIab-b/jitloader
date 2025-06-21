# jitloader/scheduler.py

import torch
from torch import nn

from .safetensor_loader import SafetensorLoader

class BaseScheduler:
    """
    Base class for all schedulers, providing common functionality.
    """
    def __init__(self, path: str, model_blueprint: nn.Module, device: str = "cuda", model_config: dict = None):
        self.device = device
        self.loader = SafetensorLoader(path, model_config=model_config)
        self.blueprint = model_blueprint.to("meta")

    def execute_layer(self, layer_name: str, *args, **kwargs):
        """
        Loads and executes a single layer of the model.
        """
        submodule = self.blueprint.get_submodule(layer_name)
        if submodule is None:
            raise ValueError(f"{layer_name} not found in blueprint")

        state_dict = {}
        for pname, _ in submodule.named_parameters(recurse=False):
            full_name = f"{layer_name}.{pname}"
            state_dict[pname] = self.loader.load_tensor(full_name, device=self.device)

        for bname, _ in submodule.named_buffers(recurse=False):
            full_name = f"{layer_name}.{bname}"
            try:
                state_dict[bname] = self.loader.load_tensor(full_name, device=self.device)
            except KeyError:
                pass

        submodule = submodule.to(self.device)
        submodule.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            output = submodule(*args, **kwargs)

        del state_dict
        submodule.to("cpu")
        torch.cuda.empty_cache()
        return output

class FluxScheduler(BaseScheduler):
    """
    Scheduler for the main FLUX transformer model.
    """
    def run_inference(self, x, timestep, context, y=None, guidance=None, **kwargs):
        from comfy.ldm.flux.layers import timestep_embedding
        from einops import rearrange, repeat
        import comfy.ldm.common_dit

        bs, c, h, w = x.shape
        patch_size = self.blueprint.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len, w_len = h // patch_size, w // patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        
        if y is None:
            y = torch.zeros((x.shape[0], self.blueprint.params.vec_in_dim), device=x.device, dtype=x.dtype)

        img = self.execute_layer('img_in', img)
        vec = self.execute_layer('time_in', timestep_embedding(timestep, 256).to(img.dtype))
        if self.blueprint.params.guidance_embed and guidance is not None:
            vec = vec + self.execute_layer('guidance_in', timestep_embedding(guidance, 256).to(img.dtype))
        vec = vec + self.execute_layer('vector_in', y[:,:self.blueprint.params.vec_in_dim])
        txt = self.execute_layer('txt_in', context)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.execute_layer('pe_embedder', ids)

        for i in range(len(self.blueprint.double_blocks)):
            img, txt = self.execute_layer(f'double_blocks.{i}', img=img, txt=txt, vec=vec, pe=pe)
        
        img = torch.cat((txt, img), 1)
        
        for i in range(len(self.blueprint.single_blocks)):
            img = self.execute_layer(f'single_blocks.{i}', img, vec=vec, pe=pe)
        
        img = img[:, txt.shape[1] :, ...]
        img = self.execute_layer('final_layer', img, vec)
        return rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=patch_size, pw=patch_size)[:,:,:h,:w]

class VAEScheduler(BaseScheduler):
    """
    Scheduler for the VAE model.
    """
    def run_decoder_inference(self, latents):
        # This is a simplified decoder implementation
        x = self.execute_layer('conv_in', latents)
        x = self.execute_layer('mid.block_1', x)
        # ... more layers would be executed here ...
        x = self.execute_layer('conv_out', x)
        return x

class T5Scheduler(BaseScheduler):
    """
    Scheduler for the T5 text encoder.
    """
    def run_encoder_inference(self, input_ids):
        hidden_states = self.execute_layer('shared', input_ids)
        for i in range(len(self.blueprint.block)):
            hidden_states = self.execute_layer(f'block.{i}', hidden_states)[0]
        hidden_states = self.execute_layer('final_layer_norm', hidden_states)
        return hidden_states

class CLIPScheduler(BaseScheduler):
    """
    Scheduler for the CLIP text encoder.
    """
    def run_encoder_inference(self, input_ids):
        hidden_states = self.execute_layer('text_model.embeddings', input_ids)
        for i in range(len(self.blueprint.text_model.encoder.layers)):
            hidden_states = self.execute_layer(f'text_model.encoder.layers.{i}', hidden_states)[0]
        hidden_states = self.execute_layer('text_model.final_layer_norm', hidden_states)
        return hidden_states