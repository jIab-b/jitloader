# jitloader/scheduler.py

import torch
from torch import nn
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from .safetensor_loader import SafetensorLoader, SAFETENSORS_DTYPE_MAP
from .mem_allocator import MemoryAllocator

class BaseScheduler:
    """
    Base class for all schedulers, providing asynchronous pre-fetching.
    """
    def __init__(self, path: str, model_blueprint: nn.Module, allocator: MemoryAllocator, device: str = "cuda", model_config: dict = None, quant_config: str = None, prefetch_depth: int = 2):
        self.device = device
        self.loader = SafetensorLoader(path, model_config=model_config, quant_config=quant_config)
        self.blueprint = model_blueprint.to("meta")
        self.allocator = allocator
        self.prefetch_depth = prefetch_depth
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch_queue = Queue(maxsize=prefetch_depth)
        self.execution_plan = []

    def _prepare_execution_plan(self, plan: list):
        self.execution_plan = plan
        for layer_name in self.execution_plan[:self.prefetch_depth]:
            future = self.executor.submit(self._load_layer_to_pool, layer_name)
            self.prefetch_queue.put((layer_name, future))

    def _load_layer_to_pool(self, layer_name: str):
        """
        Loads all tensors for a layer into pre-allocated memory pools.
        Returns a dictionary of handles (allocated tensors) and their names.
        """
        self.allocator.reset('cpu')
        handles = {}
        submodule = self.blueprint.get_submodule(layer_name)

        for name, param in submodule.named_parameters(recurse=False):
            full_name = f"{layer_name}.{name}"
            info = self.loader.get_tensor_info(full_name)
            dtype = SAFETENSORS_DTYPE_MAP[info['dtype']]
            size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(info['shape']))

            cpu_buffer = self.allocator.allocate(size, 'cpu').view(dtype).reshape(param.shape)
            self.loader.load_tensor_into(full_name, cpu_buffer)
            handles[name] = cpu_buffer

        for name, buffer in submodule.named_buffers(recurse=False):
            full_name = f"{layer_name}.{name}"
            try:
                info = self.loader.get_tensor_info(full_name)
                dtype = SAFETENSORS_DTYPE_MAP[info['dtype']]
                size = torch.tensor([], dtype=dtype).element_size() * torch.prod(torch.tensor(info['shape']))
                
                cpu_buffer = self.allocator.allocate(size, 'cpu').view(dtype).reshape(buffer.shape)
                self.loader.load_tensor_into(full_name, cpu_buffer)
                handles[name] = cpu_buffer
            except KeyError:
                pass
        return handles

    def execute_layer(self, layer_name: str, *args, **kwargs):
        """
        Executes a single layer, using the pre-fetch buffer.
        """
        # Get the next layer from the queue
        next_layer_name, future = self.prefetch_queue.get()
        if next_layer_name != layer_name:
            raise RuntimeError(f"Execution plan mismatch: expected {layer_name}, got {next_layer_name}")

        # Load the next layer in the plan into the queue
        current_index = self.execution_plan.index(layer_name)
        next_to_prefetch_index = current_index + self.prefetch_depth
        if next_to_prefetch_index < len(self.execution_plan):
            next_layer_to_prefetch = self.execution_plan[next_to_prefetch_index]
            new_future = self.executor.submit(self._load_layer_to_pool, next_layer_to_prefetch)
            self.prefetch_queue.put((next_layer_to_prefetch, new_future))

        # Get the state dict from the completed future. Tensors are already on the correct device.
        cpu_handles = future.result()
        self.allocator.reset('gpu')
        state_dict = {}

        # Transfer to GPU and build state_dict
        for name, cpu_tensor in cpu_handles.items():
            gpu_tensor = self.allocator.allocate(cpu_tensor.nbytes, 'cuda').view(cpu_tensor.dtype).reshape(cpu_tensor.shape)
            gpu_tensor.copy_(cpu_tensor)
            state_dict[name] = gpu_tensor

        submodule = self.blueprint.get_submodule(layer_name).to_empty(device=self.device)
        submodule.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            output = submodule(*args, **kwargs)

        return output

class FluxScheduler(BaseScheduler):
    """
    Scheduler for the main FLUX transformer model.
    """
    def run_inference(self, x, timestep, context, y=None, guidance=None, **kwargs):
        plan = ['img_in', 'time_in']
        if self.blueprint.params.guidance_embed and guidance is not None:
            plan.append('guidance_in')
        plan.extend(['vector_in', 'txt_in', 'pe_embedder'])
        plan.extend([f'double_blocks.{i}' for i in range(len(self.blueprint.double_blocks))])
        plan.extend([f'single_blocks.{i}' for i in range(len(self.blueprint.single_blocks))])
        plan.append('final_layer')
        self._prepare_execution_plan(plan)
        
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
        latents = latents / self.blueprint.config.get("scaling_factor", 0.18215)
        
        plan = ['post_quant_conv']
        plan.append('decoder.conv_in')
        plan.append('decoder.mid.block_1')
        plan.append('decoder.mid.attn_1')
        plan.append('decoder.mid.block_2')

        for i in reversed(range(self.blueprint.decoder.num_resolutions)):
            for j in range(self.blueprint.decoder.num_res_blocks + 1):
                plan.append(f'decoder.up.{i}.block.{j}')
            if self.blueprint.decoder.up[i].attn:
                plan.append(f'decoder.up.{i}.attn.0')
            if i != 0:
                plan.append(f'decoder.up.{i}.upsample')

        plan.extend(['decoder.norm_out', 'decoder.conv_out'])
        self._prepare_execution_plan(plan)

        h = self.execute_layer('post_quant_conv', latents)
        h = self.execute_layer('decoder.conv_in', h)
        h = self.execute_layer('decoder.mid.block_1', h)
        h = self.execute_layer('decoder.mid.attn_1', h)
        h = self.execute_layer('decoder.mid.block_2', h)

        for i in reversed(range(self.blueprint.decoder.num_resolutions)):
            for j in range(self.blueprint.decoder.num_res_blocks + 1):
                h = self.execute_layer(f'decoder.up.{i}.block.{j}', h)
            if self.blueprint.decoder.up[i].attn:
                h = self.execute_layer(f'decoder.up.{i}.attn.0', h)
            if i != 0:
                h = self.execute_layer(f'decoder.up.{i}.upsample', h)

        h = self.execute_layer('decoder.norm_out', h)
        dec = self.execute_layer('decoder.conv_out', h)
        return dec

class T5Scheduler(BaseScheduler):
    """
    Scheduler for the T5 text encoder.
    """
    def run_encoder_inference(self, input_ids):
        # The transformer blocks and final norm are inside the 'encoder' attribute
        plan = ['shared'] + [f'encoder.block.{i}' for i in range(len(self.blueprint.encoder.block))] + ['encoder.final_layer_norm']
        self._prepare_execution_plan(plan)

        hidden_states = self.execute_layer('shared', input_ids)
        for i in range(len(self.blueprint.encoder.block)):
            hidden_states = self.execute_layer(f'encoder.block.{i}', hidden_states, use_cache=False)[0]
        hidden_states = self.execute_layer('encoder.final_layer_norm', hidden_states)
        return hidden_states

class CLIPScheduler(BaseScheduler):
    """
    Scheduler for the CLIP text encoder.
    """
    def run_encoder_inference(self, input_ids):
        plan = ['text_model.embeddings'] + [f'text_model.encoder.layers.{i}' for i in range(len(self.blueprint.text_model.encoder.layers))] + ['text_model.final_layer_norm']
        self._prepare_execution_plan(plan)

        hidden_states = self.execute_layer('text_model.embeddings', input_ids)
        for i in range(len(self.blueprint.text_model.encoder.layers)):
            hidden_states = self.execute_layer(f'text_model.encoder.layers.{i}', hidden_states)[0]
        hidden_states = self.execute_layer('text_model.final_layer_norm', hidden_states)
        return hidden_states