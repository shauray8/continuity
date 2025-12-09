import torch
import time
import functools
import inspect
import os
import numpy as np
from collections import OrderedDict

# Optional dependency for image visualization
try:
    import matplotlib.pyplot as plt
    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False
    plt = None # Define plt as None if not available


# --- George's Kinda Crappy Color Palette (reuse) ---
C = {
    "HEADER": "\033[95m", "BLUE": "\033[94m", "CYAN": "\033[96m",
    "GREEN": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m",
    "ENDC": "\033[0m", "BOLD": "\033[1m", "UNDERLINE": "\033[4m",
    "DIM": "\033[2m", "PURPLE": "\033[35m", "YELLOW": "\033[33m",
    "LIGHT_BLUE": "\033[94m", # Using the same as BLUE for simplicity
    "LIGHT_GREEN": "\033[98m",
    "LIGHT_CYAN": "\033[97m",
    "LIGHT_RED": "\033[99m",
    "LIGHT_PURPLE": "\033[95m", # Same as HEADER
    "ORANGE": "\033[38;5;208m",
    "MAGENTA": "\033[35m", # Same as PURPLE
    "TEAL": "\033[36m", # Similar to CYAN
    "OLIVE": "\033[38;5;128m",
    "MAROON": "\033[38;5;88m",
}

# --- Debugger Configuration via Environment Variables ---
DEBUG_LEVEL = int(os.environ.get("DIFFUSION_DEBUG", "0")) # 0: off, 1: basic, 2: detailed tensor, 3: +asm, 4: +gpu_mem
VISUALIZE_IMAGES = int(os.environ.get("DIFFUSION_IMAGE_TENSORS", "0")) # 1: try to visualize BCHW tensors
CALC_TENSOR_STATS = int(os.environ.get("DIFFUSION_TENSOR_STATS", "1")) # 1: calc min/max/mean/std if DEBUG>=2
VIS_SAVE_DIR = "diffusion_debug_images" # Directory to save visualizations
_vis_counter = 0 # Global counter for unique image filenames

if VISUALIZE_IMAGES and not _matplotlib_available:
    print(f"{C['WARN']}WARN: DIFFUSION_IMAGE_TENSORS=1 but matplotlib not found. Cannot visualize images.{C['ENDC']}")
    VISUALIZE_IMAGES = 0 # Disable if unavailable

if VISUALIZE_IMAGES and not os.path.exists(VIS_SAVE_DIR):
    try:
        os.makedirs(VIS_SAVE_DIR)
    except OSError as e:
        print(f"{C['FAIL']}ERROR: Could not create visualization directory '{VIS_SAVE_DIR}': {e}{C['ENDC']}")
        VISUALIZE_IMAGES = 0 # Disable if cannot create dir

# --- Helper Functions ---

def _format_tensor_size(tensor):
    """ Calculates and formats tensor size in memory. """
    if not isinstance(tensor, torch.Tensor):
        return "N/A"
    try:
        numel = tensor.numel()
        element_size = tensor.element_size()
        size_bytes = numel * element_size
        if size_bytes < 1024: return f"{size_bytes} B"
        if size_bytes < 1024**2: return f"{size_bytes/1024:.2f} KB"
        if size_bytes < 1024**3: return f"{size_bytes/1024**2:.2f} MB"
        return f"{size_bytes/1024**3:.2f} GB"
    except Exception: # Handle potential errors with custom tensor types?
        return "Error"

def _get_tensor_stats(tensor):
    """ Calculates basic tensor statistics (min, max, mean, std). Handles device placement."""
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return {"min": "N/A", "max": "N/A", "mean": "N/A", "std": "N/A"}

    # Avoid large transfers or blocking if possible, maybe sample later?
    # For now, calculate on device if possible, convert small tensors?
    # Let's just calculate directly, user beware of perf impact.
    try:
        # Ensure calculation happens without modifying original tensor requires_grad
        with torch.no_grad():
            t_min = torch.min(tensor).item()
            t_max = torch.max(tensor).item()
            t_mean = torch.mean(tensor.float()).item() # Cast to float for mean/std
            t_std = torch.std(tensor.float()).item()
        return {"min": f"{t_min:.4f}", "max": f"{t_max:.4f}", "mean": f"{t_mean:.4f}", "std": f"{t_std:.4f}"}
    except Exception as e:
        return {"min": "Calc Err", "max": "Calc Err", "mean": "Calc Err", "std": f"Calc Err: {e}"}


def _visualize_image_tensor(tensor, name="tensor"):
    """ Tries to visualize a BCHW tensor using matplotlib. Saves to file. """
    global _vis_counter
    if not _matplotlib_available or not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        return

    b, c, h, w = tensor.shape
    if c not in [1, 3, 4]: # Only handle grayscale, RGB, or RGBA
        return

    print(f"{C['PURPLE']}  Attempting visualization for '{name}' (Shape: {tensor.shape})...{C['ENDC']}")

    try:
        img_tensor = tensor[0].detach().float().cpu() # Take first image in batch

        # Normalize to [0, 1] for visualization (common heuristic)
        img_min, img_max = torch.min(img_tensor), torch.max(img_tensor)
        if img_max > img_min: # Avoid division by zero
            img_tensor = (img_tensor - img_min) / (img_max - img_min)
        else:
            img_tensor = torch.zeros_like(img_tensor) # Handle constant image

        if c == 1: # Grayscale
            img_np = img_tensor.squeeze(0).numpy() # H, W
            cmap = 'gray'
        else: # RGB or RGBA
            img_np = img_tensor.permute(1, 2, 0).numpy() # H, W, C
            cmap = None

        fig, ax = plt.subplots(figsize=(max(5, w/80), max(5, h/80))) # Adjust size based on image dims
        ax.imshow(img_np, cmap=cmap)
        ax.set_title(f"{name}\nShape: {tuple(tensor.shape)} Min: {img_min:.2f} Max: {img_max:.2f}")
        ax.axis('off')

        filename = os.path.join(VIS_SAVE_DIR, f"debug_vis_{_vis_counter:04d}_{name}.png")
        plt.savefig(filename)
        plt.close(fig) # Close the figure to free memory
        print(f"{C['PURPLE']}  Visualization saved to: {filename}{C['ENDC']}")
        _vis_counter += 1

    except Exception as e:
        print(f"{C['FAIL']}  Visualization FAILED for '{name}': {e}{C['ENDC']}")


def _log_obj_info(obj, name="<obj>", indent=2):
    """ Logs information about an object, focusing on tensors. """
    prefix = " " * indent
    if isinstance(obj, torch.Tensor):
        print(f"{prefix}{C['YELLOW']}{name}{C['ENDC']}: {C['BOLD']}Tensor{C['ENDC']} "
              f"(shape={C['GREEN']}{tuple(obj.shape)}{C['ENDC']}, "
              f"dtype={str(obj.dtype).replace('torch.','')}, "
              f"device={obj.device}, "
              f"size={_format_tensor_size(obj)})")

        if DEBUG_LEVEL >= 2:
            print(f"{prefix}  {C['DIM']}Strides: {obj.stride()}, Contiguous: {obj.is_contiguous()}{C['ENDC']}")
            if CALC_TENSOR_STATS:
                stats = _get_tensor_stats(obj)
                print(f"{prefix}  {C['DIM']}Stats: Min={stats['min']}, Max={stats['max']}, Mean={stats['mean']}, Std={stats['std']}{C['ENDC']}")

        if VISUALIZE_IMAGES and obj.ndim == 4 and obj.shape[0]>0 and obj.shape[2]>4 and obj.shape[3]>4: # Heuristic for images
            _visualize_image_tensor(obj, name=f"{name}")

    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{C['YELLOW']}{name}{C['ENDC']}: {type(obj).__name__} (len={len(obj)})")
        if DEBUG_LEVEL >= 1 and len(obj) < 10: # Avoid printing huge lists/tuples
            for i, item in enumerate(obj):
                _log_obj_info(item, name=f"{name}[{i}]", indent=indent + 2)
        elif len(obj) >= 10:
            print(f"{prefix}  {C['DIM']}(Content truncated){C['ENDC']}")

    elif isinstance(obj, dict):
        print(f"{prefix}{C['YELLOW']}{name}{C['ENDC']}: {type(obj).__name__} (keys={list(obj.keys())})")
        if DEBUG_LEVEL >= 1 and len(obj) < 10:
            for k, v in obj.items():
                _log_obj_info(v, name=f"{name}['{k}']", indent=indent+2)
        elif len(obj) >= 10:
            print(f"{prefix}  {C['DIM']}(Content truncated){C['ENDC']}")
    else:
        print(f"{prefix}{C['YELLOW']}{name}{C['ENDC']}: {type(obj).__name__} = {repr(obj)}")


# --- The Debugger Decorator ---
_debugger_call_depth = 0 # Track recursion/nesting

def diffusion_debugger(func):
    """
    Decorator inspired by tinygrad's debugging. Uses environment variables:
    - DIFFUSION_DEBUG=[0,1,2,3,4]: Controls verbosity.
        0: Off
        1: Basic entry/exit, args/return types & shapes.
        2: Detailed tensor info (strides, stats if DIFFUSION_TENSOR_STATS=1).
        3: + Attempt to show GPU assembly (CUDA only, requires setup).
        4: + Attempt to show GPU memory patterns and SM usage (requires profiling tools).
    - DIFFUSION_IMAGE_TENSORS=[0,1]: Try to visualize image-like tensors (BCHW).
    - DIFFUSION_TENSOR_STATS=[0,1]: Enable/disable potentially slow tensor stats.
    """
    if DEBUG_LEVEL == 0:
        return func # No debugging, return original function

    # Get argument names from function signature
    sig = inspect.signature(func)
    arg_names = list(sig.parameters.keys())

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _debugger_call_depth
        indent_str = "| " * _debugger_call_depth
        _debugger_call_depth += 1

        # --- Function Entry ---
        class_name = ""
        if args and hasattr(args[0], '__class__') and args[0].__class__.__name__ != 'function':
            # Basic check if first arg is 'self' or 'cls'
            if arg_names and arg_names[0] in ['self', 'cls']:
                class_name = type(args[0]).__name__ + "."

        func_qualname = f"{class_name}{func.__name__}"
        print(f"\n{indent_str}{C['HEADER']}{C['BOLD']}>>> ENTER {func_qualname} (Debug Level {DEBUG_LEVEL}){C['ENDC']}")

        # --- Log Arguments ---
        if DEBUG_LEVEL >= 1:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            print(f"{indent_str}{C['CYAN']}{C['UNDERLINE']}Arguments:{C['ENDC']}")
            if not bound_args.arguments:
                print(f"{indent_str}  (No arguments)")
            else:
                for name, value in bound_args.arguments.items():
                    _log_obj_info(value, name=name, indent=_debugger_call_depth*2)

        # --- Execute Function ---
        t_start = time.perf_counter()
        result = None
        error = None
        try:
            if DEBUG_LEVEL >= 3 and torch.cuda.is_available():
                # --- Attempt to Get GPU Assembly (Requires CUDA and might have overhead) ---
                try:
                    torch.cuda.cudart().cudaProfilerStart()
                    result = func(*args, **kwargs)
                finally:
                    torch.cuda.cudart().cudaProfilerStop()
                    print(f"{indent_str}{C['LIGHT_BLUE']}{C['BOLD']}--- GPU Assembly/Profiling (Level 3) ---{C['ENDC']}")
                    print(f"{indent_str}{C['DIM']}Note: Getting detailed GPU assembly usually requires using NVIDIA's profiling tools (e.g., Nsight Compute) and analyzing the generated reports. Direct programmatic access within Python is limited.{C['ENDC']}")
                    # In a more advanced setup, you might trigger a custom CUDA compilation and inspect the PTX or SASS.
                    # This is highly dependent on the specific function and how it utilizes CUDA.

            elif DEBUG_LEVEL >= 4 and torch.cuda.is_available():
                # --- Attempt to Get GPU Memory Patterns and SM Usage (Requires Profiling) ---
                try:
                    torch.cuda.cudart().cudaProfilerStart()
                    result = func(*args, **kwargs)
                finally:
                    torch.cuda.cudart().cudaProfilerStop()
                    print(f"{indent_str}{C['ORANGE']}{C['BOLD']}--- GPU Memory & SM Usage (Level 4) ---{C['ENDC']}")
                    print(f"{indent_str}{C['DIM']}Note: Analyzing GPU memory access patterns, SM usage, and swizzle patterns typically requires using NVIDIA's profiling tools (e.g., Nsight Systems, Nsight Compute). These tools provide detailed timelines and metrics that are not directly accessible through standard Python/PyTorch APIs.{C['ENDC']}")
                    print(f"{indent_str}{C['DIM']}Consider using these tools to gain insights into GPU behavior.{C['ENDC']}")

            else:
                result = func(*args, **kwargs)

        except Exception as e:
            error = e
            print(f"{indent_str}{C['FAIL']}{C['BOLD']}!!! EXCEPTION in {func_qualname} !!!{C['ENDC']}")
            print(f"{indent_str}{C['FAIL']}{repr(e)}{C['ENDC']}")
            # Optionally: import traceback; traceback.print_exc()
        t_end = time.perf_counter()
        duration_ms = (t_end - t_start) * 1000.0

        # --- Log Return Value ---
        if error is None and DEBUG_LEVEL >= 1:
            print(f"{indent_str}{C['CYAN']}{C['UNDERLINE']}Return Value:{C['ENDC']}")
            if result is None:
                print(f"{indent_str}  None")
            else:
                # Handle multiple return values (tuple)
                if isinstance(result, tuple):
                    _log_obj_info(result, name="<TupleReturn>", indent=_debugger_call_depth*2)
                else:
                    _log_obj_info(result, name="<ReturnValue>", indent=_debugger_call_depth*2)


        # --- Function Exit ---
        print(f"{indent_str}{C['HEADER']}{C['BOLD']}<<< EXIT {func_qualname} (Wall Time: {duration_ms:.3f} ms){C['ENDC']}{C['FAIL'] if error else ''}{' [ERRORED]' if error else ''}{C['ENDC']}")

        _debugger_call_depth -= 1

        if error:
            raise error # Re-raise the exception after logging
        return result

    return wrapper

