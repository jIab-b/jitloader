# jitloader/scheduler.py

import os
import json
from typing import Dict, Any

import torch
from torch import nn

# Import helper ops from FLUX.  Re-using these avoids code duplication and
# keeps attention/math kernels consistent across projects.
from comfy.ldm.flux import math as flux_math


# --------------------------------------------------------------------------- #
#  SAFETENSOR LOADER                                                          #
# --------------------------------------------------------------------------- #
class SafetensorLoader:
    """
    Extremely thin, *dummy* implementation of an on-demand loader.

    What works:
      • Reads the JSON header to discover tensor metadata.
      • Exposes `load_tensor()` which – for now – simply returns a
        zero-initialised tensor of the correct shape/dtype so that
        downstream code compiles.

    TODO (contributors):
      • Replace file I/O with `mmap` for zero-copy reads.
      • Actually decode raw bytes into torch tensors.
      • Insert optional on-the-fly quantisation (FP8 / INT4, etc.).
      • Add asynchronous pre-fetch to overlap I/O and compute.
    """

    def __init__(self, path: str):
        self.path = path
        self.file = open(path, "rb")                     # TODO: switch to `mmap`
        header_size = int.from_bytes(self.file.read(8), "little")
        header_bytes = self.file.read(header_size)
        self.header: Dict[str, Any] = json.loads(header_bytes)
        self.tensor_meta = self.header.get("tensors", {})  # name -> meta dict

    # ------------------------------------------------------------------ #

    def _read_raw(self, offset: int, length: int) -> bytes:
        """Return raw bytes for a single tensor (placeholder)."""
        self.file.seek(offset)
        return self.file.read(length)

    # ------------------------------------------------------------------ #

    def load_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Materialise a tensor from disk.

        CURRENT BEHAVIOUR:
            Returns an all-zero tensor for compilation purposes.

        FUTURE:
            * Decode the real bytes (`self._read_raw`).
            * Apply quantisation if requested.
            * Pin to page-locked memory for faster GPU transfers.
        """
        meta = self.tensor_meta.get(name)
        if meta is None:
            raise KeyError(f"Tensor {name} not found in safetensor file")

        dtype_str = meta["dtype"]
        shape = meta["shape"]

        # Placeholder: create zeros so shapes/dtypes propagate.
        dtype = getattr(torch, dtype_str, torch.float16)
        tensor = torch.zeros(shape, dtype=dtype, device=device)

        # TODO: decode raw bytes + quantise as required.
        return tensor


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

    def run_inference(self, *model_inputs, **kw):
        """
        Entry-point replicating `Flux.forward(...)`, but using `execute_layer`
        for every individual module.

        TODO:
          • Copy the logic from `comfy/ldm/flux/model.forward` (patch prep,
            positional ids, concatenation, etc.).
          • Feed intermediate activations between sequential `execute_layer`
            calls.
          • Support ControlNet, guidance, and masks.
        """
        raise NotImplementedError("Full inference loop is a work in progress.")


# --------------------------------------------------------------------------- #
# End of file – compilation succeeds, but full functionality is pending.      #
# --------------------------------------------------------------------------- #