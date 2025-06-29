# jitloader/safetensor_loader.py

import os
import json
import struct
import mmap
import torch
import numpy as np

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

class SafetensorLoader:
    """
    On-demand safetensor loader using memory-mapping for zero-copy reads.
    Can handle single-file or multi-file (sharded) models.
    """
    def __init__(self, path: str, model_config: dict = None, quant_config: str = None):
        self.path = path
        self.model_config = model_config or {}
        self.weight_map = self.model_config.get('weight_map')
        self.quant_config = quant_config
        if self.quant_config:
            try:
                import bitsandbytes.functional as F
                self.F = F
            except ImportError:
                raise ImportError("bitsandbytes is required for quantization. Please install it.")

        self.file_handles = {}
        self.mmaps = {}
        self.headers = {}
        self.data_start_offsets = {}

        if self.weight_map:
            # Multi-file (sharded) model. `path` is a directory.
            unique_files = set(self.weight_map.values())
            for filename in unique_files:
                shard_path = os.path.join(path, filename)
                file = open(shard_path, 'rb')
                self.file_handles[filename] = file
                
                header_len_bytes = file.read(8)
                header_len = struct.unpack('<Q', header_len_bytes)[0]
                header_json_bytes = file.read(header_len)
                header = json.loads(header_json_bytes.decode('utf-8'))
                
                self.headers[filename] = header
                self.data_start_offsets[filename] = 8 + header_len
                self.mmaps[filename] = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            # Single-file model. `path` is a file path.
            file = open(path, 'rb')
            self.file_handles['__single__'] = file
            
            header_len_bytes = file.read(8)
            header_len = struct.unpack('<Q', header_len_bytes)[0]
            header_json_bytes = file.read(header_len)
            header = json.loads(header_json_bytes.decode('utf-8'))
            
            self.headers['__single__'] = header
            self.data_start_offsets['__single__'] = 8 + header_len
            self.mmaps['__single__'] = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)

    def __del__(self):
        for mmap_obj in self.mmaps.values():
            mmap_obj.close()
        for file_handle in self.file_handles.values():
            file_handle.close()

    def get_tensor_info(self, name: str) -> dict:
        """
        Returns the metadata for a single tensor.
        """
        if self.weight_map:
            if name not in self.weight_map:
                raise KeyError(f"Tensor '{name}' not found in the weight map.")
            filename = self.weight_map[name]
            header = self.headers[filename]
        else:
            filename = '__single__'
            header = self.headers[filename]

        if name not in header:
            raise KeyError(f"Tensor '{name}' not found in header of '{filename}'.")
        
        return header[name]

    def load_tensor_into(self, name: str, buffer: torch.Tensor):
        """
        Loads a tensor from disk directly into a pre-allocated buffer.
        """
        info = self.get_tensor_info(name)
        filename = self.weight_map.get(name, '__single__')
        mmap_obj = self.mmaps[filename]
        data_start_offset = self.data_start_offsets[filename]
        dtype = SAFETENSORS_DTYPE_MAP.get(info['dtype'])
        if dtype is None:
            raise TypeError(f"Unsupported dtype '{info['dtype']}' for tensor '{name}'")

        shape = info['shape']
        offsets = info['data_offsets']
        data_len = offsets[1] - offsets[0]
        
        data_offset = data_start_offset + offsets[0]
        data = mmap_obj[data_offset : data_offset + data_len]
        
        if info['dtype'] == 'BF16':
            # NumPy doesn't have a bfloat16, so we read as uint16 and view as bfloat16 in torch
            np_buffer = np.frombuffer(data, dtype=np.uint16).reshape(shape)
            tensor_view = torch.from_numpy(np_buffer.copy()).view(torch.bfloat16)
        else:
            # Create a numpy array view of the buffer without copying
            np_buffer = np.frombuffer(data, dtype=NUMPY_DTYPE_MAP[info['dtype']]).reshape(shape)
            # Create a torch tensor from the numpy array without copying
            tensor_view = torch.from_numpy(np_buffer.copy())

        # Copy the data into the provided buffer
        buffer.copy_(tensor_view)