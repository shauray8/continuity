import torch
import torch.nn as nn
import time

class DiagonalGaussianOriginal(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean

class _DiagonalGaussianOptimized(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim
        self.noise = None  # Pre-allocated tensor for reuse

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp_(0.5 * logvar)  # In-place operation
            if self.noise is None or self.noise.shape != mean.shape:
                self.noise = torch.randn_like(mean, memory_format=torch.preserve_format)
            return mean + std * self.noise  # Reuse pre-allocated noise
        return mean

import torch.cuda

class DiagonalGaussianOptimized(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim
        self.stream = torch.cuda.Stream()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)  # Use exp (not in-place to avoid blocking)
            with torch.cuda.stream(self.stream):
                noise = torch.randn_like(mean, memory_format=torch.preserve_format)
                return mean + std * noise
        return mean

import torch
import time

# Function-based implementation
def diagonal_gaussian(z: torch.Tensor, sample: bool = True, chunk_dim: int = 1) -> torch.Tensor:
    mean, logvar = torch.chunk(z, 2, dim=chunk_dim)
    if sample:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)
    return mean

# Benchmark
z = torch.randn(1024, 512)

# Timing the function
start = time.time()
for _ in range(1000):
    output = diagonal_gaussian(z)
end = time.time()
print(f"Function time: {end - start:.5f} seconds")

# Timing the class-based implementation
model = DiagonalGaussianOptimized()
start = time.time()
for _ in range(1000):
    output = model(z)
end = time.time()
print(f"Class-based time: {end - start:.5f} seconds")

"""
def benchmark_model(model, z, iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        output = model(z)
    return time.time() - start_time

iterations = 1000
z_size = (2048, 1024)  # Adjust based on your system memory
z = torch.randn(z_size, device="cuda")  # Using GPU for benchmarking

# Initialize models
original_model = DiagonalGaussianOriginal().cuda()
optimized_model = DiagonalGaussianOptimized().cuda()
_optimized_model = _DiagonalGaussianOptimized().cuda()

# Benchmark original model
original_time = benchmark_model(original_model, z, iterations)

# Benchmark optimized model
optimized_time = benchmark_model(optimized_model, z, iterations)
_optimized_time = benchmark_model(_optimized_model, z, iterations)

# Compare results
print(f"Original model time: {original_time:.4f} seconds")
print(f"Stream Optimized model time: {optimized_time:.4f} seconds")
print(f"Optimized model time: {_optimized_time:.4f} seconds")
"""
