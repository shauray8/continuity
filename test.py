import torch 
import torch._dynamo
torch._dynamo.config.suppress_errors = True


@torch.compile(mode="reduce-overhead")
def foo(x):
    # GRAPH 1
    y = x * x * x
    # graph break triggered here
    if y.sum() > 0:
        # GRAPH 2
        z = y ** y
    else:
        # GRAPH 3
        z = (y.abs() ** y.abs())
    torch._dynamo.graph_break()
    # GRAPH 4
    return z * torch.rand_like(z)

# the first run warms up each graph, which does things like CuBlas or Triton benchmarking
foo(torch.arange(0, 10, device="cuda"))
# The second run does a CUDA Graph recording, and replays it
foo(torch.arange(0, 10, device="cuda"))
# Finally we hit the optimized, CUDA Graph replay path
foo(torch.arange(0, 10, device="cuda"))
