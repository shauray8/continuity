import time
import os
import inspect
import torch
import threading
import atexit
from enum import Enum, auto
from typing import Optional, Dict, List, Any, Set, Callable
from contextlib import contextmanager
import numpy as np
from dataclasses import dataclass
import traceback
from functools import wraps

# ANSI color codes - very hotz-like to use raw color codes
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    BOLD = "\033[1m"

class LogLevel(Enum):
    SILENT = auto()
    ERROR = auto()
    WARN = auto()
    INFO = auto()
    DEBUG = auto()
    TRACE = auto()

@dataclass
class TimedEvent:
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent: Optional['TimedEvent'] = None
    children: List['TimedEvent'] = None
    metadata: Dict[str, Any] = None
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def duration(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def add_child(self, child: 'TimedEvent') -> None:
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)


class DiffusionLogger:
    """
    High-performance logger for diffusion model inference with inspiration from tinygrad.
    
    Designed to track and display detailed performance metrics with minimal overhead,
    following a philosophy similar to George Hotz's approach in tinygrad.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DiffusionLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, 
                 level: LogLevel = LogLevel.INFO,
                 file_path: Optional[str] = None,
                 color_enabled: bool = True,
                 timing_enabled: bool = True,
                 memory_tracking: bool = True,
                 torch_ops_tracking: bool = True):
        
        if getattr(self, '_initialized', False):
            return
            
        self._initialized = True
        self.level = level
        self.file_path = file_path
        self.color_enabled = color_enabled
        self.timing_enabled = timing_enabled
        self.memory_tracking = memory_tracking
        self.torch_ops_tracking = torch_ops_tracking
        
        # For timing hierarchical operations
        self.current_event = None
        self.root_events = []
        
        # Track memory usage
        self.peak_memory = 0
        self.mem_snapshots = []
        
        # Op counters
        self.op_counters = {}
        self.torch_op_times = {}
        
        # Set to track registered torch ops
        self.registered_ops = set()
        
        # Initialize file if needed
        if self.file_path:
            with open(self.file_path, 'w') as f:
                f.write(f"=== DIFFUSION INFERENCE LOG: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
        # Hook into torch operations if enabled
        if self.torch_ops_tracking:
            self._register_torch_hooks()
        
        # Clean up on exit
        atexit.register(self._cleanup)
    
    def _cleanup(self):
        """Clean up resources and print final stats on program exit."""
        if len(self.root_events) > 0:
            self.print_final_stats()
    
    def _register_torch_hooks(self):
        """Register hooks to track torch operations."""
        # Hook common tensor operations instead of trying to hook nn.Module
        # This is a more reliable approach and avoids the AttributeError
        for op_name in ['add', 'sub', 'mul', 'div', 'matmul', 'bmm', 'conv2d']:
            if hasattr(torch.Tensor, op_name) and op_name not in self.registered_ops:
                self._patch_torch_op(torch.Tensor, op_name)
                self.registered_ops.add(op_name)
                
        # Hook common functional operations
        for op_name in ['linear', 'conv2d', 'layer_norm', 'softmax']:
            if hasattr(torch.nn.functional, op_name) and op_name not in self.registered_ops:
                op_func = getattr(torch.nn.functional, op_name)
                patched_func = self._create_patched_func(op_func, f"nn.functional.{op_name}")
                setattr(torch.nn.functional, op_name, patched_func)
                self.registered_ops.add(op_name)
        
        # Register forward hooks for common modules
        def register_hooks_for_module_classes():
            """Register hooks for common PyTorch module classes."""
            module_classes = [
                torch.nn.Conv2d,
                torch.nn.Linear,
                torch.nn.LayerNorm,
                torch.nn.MultiheadAttention,
                torch.nn.TransformerEncoderLayer,
                torch.nn.TransformerDecoderLayer
            ]
            
            for module_class in module_classes:
                # Patch the forward method
                original_forward = module_class.forward
                
                @wraps(original_forward)
                def patched_forward(self, *args, **kwargs):
                    logger = DiffusionLogger()
                    module_name = self.__class__.__name__
                    
                    # Log module call
                    with logger.operation_context(f"nn.{module_name}"):
                        result = original_forward(self, *args, **kwargs)
                        
                        # Log memory for result tensor if applicable
                        if isinstance(result, torch.Tensor) and logger.memory_tracking:
                            mem_size = result.element_size() * result.nelement()
                            if mem_size > 1024 * 1024:  # Only log if > 1MB
                                logger.log_trace(
                                    f"🧠 {module_name} output tensor: {tuple(result.shape)}, "
                                    f"{logger._format_bytes(mem_size)}"
                                )
                        
                        return result
                
                # Apply the patched method
                module_class.forward = patched_forward
        
        # Register hooks for module classes
        try:
            register_hooks_for_module_classes()
        except Exception as e:
            # Fail gracefully if module hooking fails
            self.log_warn(f"Failed to register module hooks: {str(e)}")
    
    def _create_patched_func(self, original_func, func_name):
        """Create a patched version of a function that logs its execution."""
        @wraps(original_func)
        def patched_func(*args, **kwargs):
            logger = DiffusionLogger()
            
            # Track operation count
            if func_name not in logger.op_counters:
                logger.op_counters[func_name] = 0
            logger.op_counters[func_name] += 1
            
            # Time the operation
            start_time = time.time()
            try:
                result = original_func(*args, **kwargs)
                
                # Trace log for large tensors
                if isinstance(result, torch.Tensor) and logger.memory_tracking:
                    mem_size = result.element_size() * result.nelement()
                    if mem_size > 10 * 1024 * 1024:  # Only trace log if > 10MB
                        logger.log_trace(
                            f"🧠 {func_name} produced tensor of shape {tuple(result.shape)}, "
                            f"using {logger._format_bytes(mem_size)}"
                        )
                        
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                # Track time stats
                if func_name not in logger.torch_op_times:
                    logger.torch_op_times[func_name] = []
                logger.torch_op_times[func_name].append(duration)
                
                # Trace log slow operations
                if duration > 0.01:  # 10ms threshold for trace logging
                    logger.log_trace(f"⏱️ {func_name} took {duration*1000:.2f}ms")
                
                # Debug log extremely slow operations
                if duration > 0.1:  # 100ms threshold for "slow" operations
                    logger.log_debug(f"⚠️ Slow operation: {func_name} took {duration*1000:.2f}ms")
        
        return patched_func
    
    def _patch_torch_op(self, cls, op_name):
        """Patch a torch operation to log its usage."""
        if hasattr(cls, op_name):
            original_op = getattr(cls, op_name)
            setattr(cls, f"_original_{op_name}", original_op)
            
            @wraps(original_op)
            def wrapped_op(*args, **kwargs):
                logger = DiffusionLogger()
                op_full_name = f"{cls.__name__}.{op_name}"
                
                # Track operation count
                if op_full_name not in logger.op_counters:
                    logger.op_counters[op_full_name] = 0
                logger.op_counters[op_full_name] += 1
                
                # Only time operations if we're in TRACE mode to reduce overhead
                if logger._should_log(LogLevel.TRACE):
                    start_time = time.time()
                    try:
                        result = original_op(*args, **kwargs)
                        
                        # Only log large tensor results to reduce noise
                        if isinstance(result, torch.Tensor) and logger.memory_tracking:
                            mem_size = result.element_size() * result.nelement()
                            if mem_size > 50 * 1024 * 1024:  # Only log if > 50MB
                                logger.log_trace(
                                    f"🧠 {op_name} produced tensor of shape {tuple(result.shape)}, "
                                    f"using {logger._format_bytes(mem_size)}"
                                )
                                
                        return result
                    finally:
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Track time stats
                        if op_full_name not in logger.torch_op_times:
                            logger.torch_op_times[op_full_name] = []
                        logger.torch_op_times[op_full_name].append(duration)
                        
                        # Log extremely slow operations
                        if duration > 0.1:  # 100ms threshold for "slow" operations
                            logger.log_debug(f"⚠️ Slow operation: {op_full_name} took {duration*1000:.2f}ms")
                else:
                    # Skip timing if not in TRACE mode
                    return original_op(*args, **kwargs)
            
            # Apply the wrapped operation
            try:
                setattr(cls, op_name, wrapped_op)
            except (AttributeError, TypeError):
                # Some operations might be read-only properties or methods
                # Skip these and log a warning
                self.log_warn(f"Failed to patch operation: {op_name} on {cls.__name__}")
                if op_name in self.registered_ops:
                    self.registered_ops.remove(op_name)
    
    def enable(self):
        """Enable the logger."""
        self.level = LogLevel.INFO
    
    def disable(self):
        """Disable the logger completely."""
        self.level = LogLevel.SILENT
    
    def set_level(self, level: LogLevel):
        """Set the logging level."""
        self.level = level
    
    def _should_log(self, level: LogLevel) -> bool:
        """Check if we should log at this level."""
        return self.level.value >= level.value
    
    def _format_bytes(self, size_bytes: int) -> str:
        """Format bytes to human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.1f}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.1f}MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.1f}GB"
    
    def _get_color_for_duration(self, duration: float) -> str:
        """Return appropriate color based on duration."""
        if not self.color_enabled:
            return ""
        
        if duration < 0.001:  # < 1ms
            return Colors.GREEN
        elif duration < 0.01:  # < 10ms
            return Colors.BLUE
        elif duration < 0.1:   # < 100ms
            return Colors.YELLOW
        elif duration < 1.0:   # < 1s
            return Colors.MAGENTA
        else:                  # >= 1s
            return Colors.RED
    
    def _get_color_for_memory(self, bytes_used: int) -> str:
        """Return appropriate color based on memory usage."""
        if not self.color_enabled:
            return ""
        
        if bytes_used < 1024 * 1024:  # < 1MB
            return Colors.GREEN
        elif bytes_used < 10 * 1024 * 1024:  # < 10MB
            return Colors.BLUE
        elif bytes_used < 100 * 1024 * 1024:  # < 100MB
            return Colors.YELLOW
        elif bytes_used < 1024 * 1024 * 1024:  # < 1GB
            return Colors.MAGENTA
        else:  # >= 1GB
            return Colors.RED
    
    def _get_color_for_level(self, level: LogLevel) -> str:
        """Return appropriate color based on log level."""
        if not self.color_enabled:
            return ""
        
        if level == LogLevel.ERROR:
            return Colors.RED
        elif level == LogLevel.WARN:
            return Colors.YELLOW
        elif level == LogLevel.INFO:
            return Colors.BRIGHT_WHITE
        elif level == LogLevel.DEBUG:
            return Colors.BRIGHT_BLACK
        elif level == LogLevel.TRACE:
            return Colors.BRIGHT_BLACK
        else:
            return ""
    
    def _log(self, level: LogLevel, msg: str, metadata: Dict[str, Any] = None):
        """Main internal logging method."""
        if not self._should_log(level):
            return
        
        # Get caller info
        frame = inspect.currentframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        
        # Format the message with colors
        color = self._get_color_for_level(level)
        level_str = f"{level.name:<5}"
        
        # Create indent based on current event depth
        indent = ""
        if self.current_event:
            indent = "  " * self.current_event.depth
        
        timestamp = time.strftime("%H:%M:%S")
        location = f"{filename}:{lineno}"
        
        formatted_msg = (
            f"{color}{timestamp} {level_str} "
            f"[{location}] {indent}{msg}{Colors.RESET}"
        )
        
        # Print to terminal
        print(formatted_msg)
        
        # Write to file if enabled
        if self.file_path:
            # Remove color codes for file output
            clean_msg = msg
            for color_code in vars(Colors).values():
                if isinstance(color_code, str) and color_code.startswith("\033"):
                    clean_msg = clean_msg.replace(color_code, "")
            
            with open(self.file_path, 'a') as f:
                f.write(f"{timestamp} {level_str} [{location}] {indent}{clean_msg}\n")
        
        # Store metadata if timing is enabled
        if self.timing_enabled and self.current_event and metadata:
            self.current_event.metadata.update(metadata)
    
    def log_error(self, msg: str, metadata: Dict[str, Any] = None):
        """Log an error message."""
        self._log(LogLevel.ERROR, msg, metadata)
    
    def log_warn(self, msg: str, metadata: Dict[str, Any] = None):
        """Log a warning message."""
        self._log(LogLevel.WARN, msg, metadata)
    
    def log_info(self, msg: str, metadata: Dict[str, Any] = None):
        """Log an info message."""
        self._log(LogLevel.INFO, msg, metadata)
    
    def log_debug(self, msg: str, metadata: Dict[str, Any] = None):
        """Log a debug message."""
        self._log(LogLevel.DEBUG, msg, metadata)
    
    def log_trace(self, msg: str, metadata: Dict[str, Any] = None):
        """Log a trace message (very detailed)."""
        self._log(LogLevel.TRACE, msg, metadata)
    
    @contextmanager
    def operation_context(self, name: str, metadata: Dict[str, Any] = None):
        """Context manager to time and track an operation."""
        if not self.timing_enabled:
            yield
            return
            
        start_time = time.time()
        
        # Create new event
        event = TimedEvent(name=name, start_time=start_time, metadata=metadata or {})
        
        # Track memory before
        if self.memory_tracking and torch.cuda.is_available():
            try:
                mem_before = torch.cuda.memory_allocated()
                event.metadata['mem_before'] = mem_before
            except Exception:
                # Handle case where CUDA is "available" but not actually working
                event.metadata['mem_before'] = 0
        
        # Set parent-child relationship
        old_event = self.current_event
        if old_event:
            old_event.add_child(event)
        else:
            self.root_events.append(event)
        
        self.current_event = event
        
        try:
            self.log_debug(f"▶️ Starting: {name}")
            yield
        except Exception as e:
            self.log_error(f"💥 Exception in {name}: {str(e)}")
            event.metadata['error'] = str(e)
            event.metadata['traceback'] = traceback.format_exc()
            raise
        finally:
            end_time = time.time()
            event.end_time = end_time
            duration = end_time - start_time
            
            # Track memory after
            if self.memory_tracking and torch.cuda.is_available():
                try:
                    mem_after = torch.cuda.memory_allocated()
                    event.metadata['mem_after'] = mem_after
                    mem_diff = mem_after - event.metadata.get('mem_before', 0)
                    
                    # Update peak memory
                    if mem_after > self.peak_memory:
                        self.peak_memory = mem_after
                    
                    # Color based on memory change
                    mem_color = self._get_color_for_memory(abs(mem_diff))
                    mem_diff_str = f"{self._format_bytes(mem_diff)}"
                    if mem_diff > 0:
                        mem_diff_str = f"+{mem_diff_str}"
                    
                    mem_str = f" [Mem: {mem_color}{mem_diff_str}{Colors.RESET}]"
                except Exception:
                    mem_str = ""
            else:
                mem_str = ""
            
            # Log the completion with duration
            duration_color = self._get_color_for_duration(duration)
            self.log_debug(
                f"✓ Completed: {name} in {duration_color}{duration*1000:.2f}ms{Colors.RESET}{mem_str}"
            )
            
            # Restore previous event
            self.current_event = old_event
    
    def track_function(self, func=None, *, name=None):
        """Decorator to track function execution time and details."""
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                func_name = name or f.__qualname__
                with self.operation_context(func_name):
                    return f(*args, **kwargs)
            return wrapper
        
        if func is None:
            return decorator
        return decorator(func)
    
    def memory_snapshot(self, label: str = None):
        """Take a snapshot of current memory usage."""
        if not self.memory_tracking:
            return
        
        try:
            current_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            snapshot = {
                'time': time.time(),
                'memory': current_mem,
                'label': label or f"Snapshot {len(self.mem_snapshots) + 1}"
            }
            
            self.mem_snapshots.append(snapshot)
            self.log_info(
                f"📸 Memory snapshot '{snapshot['label']}': "
                f"{self._get_color_for_memory(current_mem)}{self._format_bytes(current_mem)}{Colors.RESET}"
            )
            
            if current_mem > self.peak_memory:
                self.peak_memory = current_mem
            
            return snapshot
        except Exception as e:
            self.log_warn(f"Failed to take memory snapshot: {str(e)}")
            return None
    
    def print_event_tree(self, event=None, depth=0):
        """Print the hierarchical tree of timed events."""
        events_to_print = []
        
        # If no event is specified, print all root events
        if event is None:
            events_to_print = self.root_events
        else:
            events_to_print = [event]
            
        for evt in events_to_print:
            # Calculate indent
            indent = "  " * depth
            
            # Get duration color
            duration_color = self._get_color_for_duration(evt.duration)
            
            # Format memory info if available
            mem_str = ""
            if 'mem_before' in evt.metadata and 'mem_after' in evt.metadata:
                mem_diff = evt.metadata['mem_after'] - evt.metadata['mem_before']
                mem_color = self._get_color_for_memory(abs(mem_diff))
                
                mem_diff_str = f"{self._format_bytes(mem_diff)}"
                if mem_diff > 0:
                    mem_diff_str = f"+{mem_diff_str}"
                elif mem_diff < 0:
                    mem_diff_str = f"-{self._format_bytes(abs(mem_diff))}"
                
                mem_str = f" {mem_color}[Mem: {mem_diff_str}]{Colors.RESET}"
            
            # Print the event
            print(
                f"{indent}• {evt.name}: "
                f"{duration_color}{evt.duration*1000:.2f}ms{Colors.RESET}{mem_str}"
            )
            
            # Recursively print children
            if evt.children:
                for child in evt.children:
                    self.print_event_tree(child, depth + 1)
    
    def print_memory_summary(self):
        """Print a summary of memory usage."""
        if not self.memory_tracking or not self.mem_snapshots:
            print("Memory tracking is disabled or no snapshots were taken.")
            return
        
        print("\n=== MEMORY USAGE SUMMARY ===")
        print(f"Peak memory: {self._get_color_for_memory(self.peak_memory)}"
              f"{self._format_bytes(self.peak_memory)}{Colors.RESET}")
        
        print("\nSnapshots:")
        baseline = self.mem_snapshots[0]['memory'] if self.mem_snapshots else 0
        
        for i, snapshot in enumerate(self.mem_snapshots):
            mem = snapshot['memory']
            diff_from_baseline = mem - baseline
            diff_str = ""
            
            if i > 0:
                diff_from_prev = mem - self.mem_snapshots[i-1]['memory']
                diff_color = self._get_color_for_memory(abs(diff_from_prev))
                
                if diff_from_prev > 0:
                    diff_str = f" ({diff_color}+{self._format_bytes(diff_from_prev)}{Colors.RESET} from previous)"
                elif diff_from_prev < 0:
                    diff_str = f" ({diff_color}-{self._format_bytes(abs(diff_from_prev))}{Colors.RESET} from previous)"
            
            baseline_diff = ""
            if i > 0:
                baseline_color = self._get_color_for_memory(abs(diff_from_baseline))
                baseline_diff = f" [{baseline_color}{'+' if diff_from_baseline >= 0 else '-'}" \
                                f"{self._format_bytes(abs(diff_from_baseline))}{Colors.RESET} from baseline]"
            
            mem_color = self._get_color_for_memory(mem)
            print(
                f"{i+1}. {snapshot['label']}: "
                f"{mem_color}{self._format_bytes(mem)}{Colors.RESET}{diff_str}{baseline_diff}"
            )
    
    def print_op_statistics(self):
        """Print statistics about operations."""
        if not self.op_counters:
            print("No operations were tracked.")
            return
        
        print("\n=== OPERATION STATISTICS ===")
        
        # Sort operations by count (most frequent first)
        sorted_ops = sorted(self.op_counters.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Operation':<40} {'Count':<10} {'Avg Time (ms)':<15}")
        print("-" * 65)
        
        for op_name, count in sorted_ops:
            avg_time = 0
            if op_name in self.torch_op_times and self.torch_op_times[op_name]:
                avg_time = np.mean(self.torch_op_times[op_name]) * 1000  # Convert to ms
            
            time_color = self._get_color_for_duration(avg_time / 1000)  # Convert ms to seconds for color
            
            print(f"{op_name:<40} {count:<10} {time_color}{avg_time:.2f}{Colors.RESET}")
    
    def print_final_stats(self):
        """Print comprehensive final statistics."""
        print("\n" + "=" * 50)
        print("DIFFUSION INFERENCE EXECUTION SUMMARY")
        print("=" * 50)
        
        # Print timing tree
        print("\n--- OPERATION TIMING TREE ---")
        for root_event in self.root_events:
            self.print_event_tree(root_event)
        
        # Print memory stats if enabled
        if self.memory_tracking:
            self.print_memory_summary()
        
        # Print operation stats if any were tracked
        if self.op_counters:
            self.print_op_statistics()
        
        print("\n" + "=" * 50)
