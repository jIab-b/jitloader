import json
import struct
import torch
import numpy as np

SAFETENSORS_DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

NUMPY_DTYPE_MAP = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

class TensorReader:
    """
    A class to read tensors from a .safetensors file without loading the
    entire file into memory. It reads tensor data from the file by seeking
    to the correct offset.
    """
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            header_len_bytes = f.read(8)
            header_len = struct.unpack('<Q', header_len_bytes)[0]
            header_json_bytes = f.read(header_len)
            self.header = json.loads(header_json_bytes.decode('utf-8'))
            self.data_start_offset = 8 + header_len

    def keys(self):
        """Returns the names of the tensors in the file."""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key):
        """
        Reads and returns a single tensor specified by its key.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file.")

        info = self.header[key]
        dtype = SAFETENSORS_DTYPE_MAP.get(info['dtype'])
        if dtype is None:
            raise TypeError(f"Unsupported dtype '{info['dtype']}' for tensor '{key}'")
        
        shape = info['shape']
        offsets = info['data_offsets']
        data_len = offsets[1] - offsets[0]

        with open(self.path, 'rb') as f:
            f.seek(self.data_start_offset + offsets[0])
            data = f.read(data_len)
            
            if info['dtype'] == 'BF16':
                np_array = np.frombuffer(data, dtype=np.uint16).reshape(shape)
                tensor = torch.from_numpy(np_array).view(torch.bfloat16)
            else:
                numpy_dtype = NUMPY_DTYPE_MAP.get(info['dtype'])
                if numpy_dtype is None:
                    raise TypeError(f"Unsupported numpy dtype '{info['dtype']}' for tensor '{key}'")
                np_array = np.frombuffer(data, dtype=numpy_dtype).reshape(shape)
                tensor = torch.from_numpy(np_array)
        
        return tensor

def safe_open(path, framework="pt", device="cpu"):
    """
    Mimics the safetensors.torch.safe_open function but uses the memory-efficient
    TensorReader. The 'framework' and 'device' arguments are ignored.
    """
    return TensorReader(path)