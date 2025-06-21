# jitloader/mem_allocator.py

import torch

class MemoryAllocator:
    """
    A simple bump-allocator for managing pre-allocated CPU and GPU memory pools.
    It allocates memory sequentially and is reset after each layer's execution.
    """
    def __init__(self, cpu_pool_size: int, gpu_pool_size: int, device: str = "cuda"):
        self.device = device
        self.cpu_pool = torch.zeros(cpu_pool_size, dtype=torch.uint8, device='cpu')
        self.gpu_pool = torch.zeros(gpu_pool_size, dtype=torch.uint8, device=self.device)
        self.cpu_offset = 0
        self.gpu_offset = 0

    def allocate(self, size: int, device: str):
        """
        Allocates a block of memory of a given size from the specified pool.
        """
        size = int(size)
        if device == 'cpu':
            if self.cpu_offset + size > self.cpu_pool.numel():
                raise MemoryError(f"CPU memory pool out of memory. Requested {size}, available {self.cpu_pool.numel() - self.cpu_offset}")
            tensor_slice = self.cpu_pool.narrow(0, self.cpu_offset, size)
            self.cpu_offset += size
            return tensor_slice
        elif device == 'cuda':
            if self.gpu_offset + size > self.gpu_pool.numel():
                raise MemoryError(f"GPU memory pool out of memory. Requested {size}, available {self.gpu_pool.numel() - self.gpu_offset}")
            tensor_slice = self.gpu_pool.narrow(0, self.gpu_offset, size)
            self.gpu_offset += size
            return tensor_slice
        else:
            raise ValueError(f"Unknown device for allocation: {device}")

    def reset(self, device: str = None):
        """
        Resets the offset for a specific pool or all pools, making them fully available.
        """
        if device is None or device == 'cpu':
            self.cpu_offset = 0
        if device is None or device == 'cuda':
            self.gpu_offset = 0