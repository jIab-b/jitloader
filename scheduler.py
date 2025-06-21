# jitloader/scheduler.py

import os
import json
import struct
import mmap
from typing import Dict, Any

import torch
from torch import nn
import numpy as np

# Import helper ops from FLUX.  Re-using these avoids code duplication and
# keeps attention/math kernels consistent across projects.
from comfy.ldm.flux import math as flux_math


SAFETENSORS_DTYPE_MAP = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}

NUMPY_DTYPE_MAP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16,
    "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
}


# --------------------------------------------------------------------------- #
#  SAFETENSOR LOADER                                                          #
# --------------------------------------------------------------------------- #
class SafetensorLoader:
    """
    On-demand safetensor loader using memory-mapping for zero-copy reads.
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, 'rb') as f:
            header_len_bytes = f.read(8)
            header_len = struct.unpack('<Q', header_len_bytes)[0]
            header_json_bytes = f.read(header_len)
            self.header = json.loads(header_json_bytes.decode('utf-8'))
            self.data_start_offset = 8 + header_len
        
        self.file = open(path, "rb")
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def __del__(self):
        if hasattr(self, 'mmap') and self.mmap:
            self.mmap.close()
        if hasattr(self, 'file') and self.file:
            self.file.close()

    def load_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Materialise a tensor from disk.
        """
        if name not in self.header:
            # Fallback for nested tensor metadata
            if "tensors" in self.header and name in self.header["tensors"]:
                info = self.header["tensors"][name]
            else:
                raise KeyError(f"Tensor '{name}' not found in the file.")
        else:
            info = self.header[name]

        dtype = SAFETENSORS_DTYPE_MAP.get(info['dtype'])
        if dtype is None:
            raise TypeError(f"Unsupported dtype '{info['dtype']}' for tensor '{name}'")

        shape = info['shape']
        offsets = info['data_offsets']
        data_len = offsets[1] - offsets[0]
        
        data_offset = self.data_start_offset + offsets[0]
        data = self.mmap[data_offset : data_offset + data_len]

        if info['dtype'] == 'BF16':
            np_array = np.frombuffer(data, dtype=np.uint16).reshape(shape)
            tensor = torch.from_numpy(np_array).view(torch.bfloat16)
        else:
            numpy_dtype = NUMPY_DTYPE_MAP.get(info['dtype'])
            if numpy_dtype is None:
                raise TypeError(f"Unsupported numpy dtype '{info['dtype']}' for tensor '{name}'")
            np_array = np.frombuffer(data, dtype=numpy_dtype).reshape(shape)
            tensor = torch.from_numpy(np_array)
        
        return tensor.to(device)


# --------------------------------------------------------------------------- #
#  INFERENCE SCHEDULER                                                        #
# --------------------------------------------------------------------------- #
class InferenceScheduler:
    """
    Orchestrates Flux inference by loading exactly one layer at a time.

    The high-level control flow mirrors `comfy/ldm/flux/model.forward`.
    All heavyweight tasks (weight streaming, quantisation, TE kernels)
    are stubbed with TODO markers.
    """

    def __init__(self, safetensor_path: str, model_blueprint: nn.Module, device: str = "cuda"):
        self.device = device
        self.loader = SafetensorLoader(safetensor_path)
        # Place the blueprint on the 'meta' device to avoid weight allocation.
        self.blueprint = model_blueprint.to("meta")

    # ------------------------------------------------------------------ #

    def execute_layer(self, layer_name: str, *args, **kwargs):
        """
        1. Fetch the blueprint sub-module.
        2. Stream its weights via `SafetensorLoader`.
        3. Run the computation on `self.device`.
        4. Free GPU/CPU memory immediately after.
        """
        submodule = self.blueprint.get_submodule(layer_name)
        if submodule is None:
            raise ValueError(f"{layer_name} not found in blueprint")

        state_dict = {}

        # Load parameters (non-recursive to keep memory bounded).
        for pname, _ in submodule.named_parameters(recurse=False):
            full_name = f"{layer_name}.{pname}"
            state_dict[pname] = self.loader.load_tensor(full_name, device=self.device)

        # Some modules hold buffers (e.g., LayerNorm stats).
        for bname, _ in submodule.named_buffers(recurse=False):
            full_name = f"{layer_name}.{bname}"
            try:
                state_dict[bname] = self.loader.load_tensor(full_name, device=self.device)
            except KeyError:
                pass  # Optional buffer not present.

        submodule = submodule.to(self.device)
        submodule.load_state_dict(state_dict, strict=False)

        with torch.no_grad():
            output = submodule(*args, **kwargs)

        # Cleanup
        del state_dict
        submodule.to("cpu")
        torch.cuda.empty_cache()

        return output

    # ------------------------------------------------------------------ #

    def run_inference(self, x, timestep, context, y=None, guidance=None, **kwargs):
        """
        Entry-point replicating `Flux.forward(...)`, but using `execute_layer`
        for every individual module.
        """
        from comfy.ldm.flux.layers import timestep_embedding
        from einops import rearrange, repeat
        import comfy.ldm.common_dit

        bs, c, h, w = x.shape
        patch_size = self.blueprint.patch_size
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        
        # Replicate forward_orig logic
        if y is None:
            y = torch.zeros((x.shape[0], self.blueprint.params.vec_in_dim), device=x.device, dtype=x.dtype)

        img = self.execute_layer('img_in', img)
        vec = self.execute_layer('time_in', timestep_embedding(timestep, 256).to(img.dtype))
        if self.blueprint.params.guidance_embed:
            if guidance is not None:
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


# --------------------------------------------------------------------------- #
# End of file – compilation succeeds, but full functionality is pending.      #
# --------------------------------------------------------------------------- #