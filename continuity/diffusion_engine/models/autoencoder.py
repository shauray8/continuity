import torch
import torch.nn as nn
import time
from torch import Tensor
import triton
import triton.language as tl


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

@triton.jit
def swish_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = 1 / (1 + tl.exp(-x))
    tl.store(output_ptr + offsets, output, mask=mask)

# Kernel launch function
def launch_swish(x, output):
    BLOCK_SIZE = 1024  # or another appropriate value
    n_element=x.numel()
    grid = (n_element + BLOCK_SIZE - 1) // BLOCK_SIZE  # Calculate grid size
    swish_kernel[grid](x, output, x.numel(), BLOCK_SIZE=BLOCK_SIZE)

# Define the original ResnetBlock
class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)  # Swish activation
        launch_swish(h,h)
        h = self.conv1(h)

        h = self.norm2(h)
        launch_swish(h,h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


# Define the Triton-optimized ResnetBlock (you can use the implementation from the previous response here)

# Benchmarking function
def benchmark(model, input_tensor, num_iterations=100):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        start_time = time.time()
        for _ in range(num_iterations):
            model(input_tensor)
        end_time = time.time()
    return (end_time - start_time) / num_iterations

# Create a test dataset
batch_size = 32
input_tensor = torch.randn(batch_size, 64, 32, 32)  # Example input size
original_model = ResnetBlock(64, 128).cuda()  # Use CUDA for GPU acceleration
# Assuming TritonResnetBlock is defined as per the previous example
# triton_model = TritonResnetBlock(64, 128).cuda()  # Uncomment when Triton model is ready

# Benchmark original model
original_time = benchmark(original_model.cuda(), input_tensor.cuda())
print(f"Original ResnetBlock average time: {original_time:.6f} seconds")
