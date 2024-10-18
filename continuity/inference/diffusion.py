import itertools
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union, Callable, TypeVar, cast
)

from tqdm import tqdm
from collections import Counter

logger = init_logger(__name__)

class Diffusion:
    """
    A Diffusion module for generating images from given prompts and sampling parameters.
        This class will includes a text_encoder, a tokenizer, a scheduler, a diffusion model (possibly distributed across multiple GPUs) (not sure about that), and GPU memory space allocated for intermediate states (aka KV cache). Given a batch of prompts and sampling parameters, this class generates images from the model, using an intelligent batching mechanism and efficient memory management.

        Args:
            ## Basic required parameters ## 
            model: The diffusion model (e.g., FLUX or other variants) that defines the architecture used for the flow-matching process.
            lora_adapters: Low-Rank Adaptation (LoRA) layers to fine-tune the model on specific tasks while keeping memory and computation requirements low..
            controlnet: If using specific conditional features like ControlNet to guide diffusion, this parameter can include conditions such as depth maps, semantic segments, etc.
            use_memory_efficient_attention: This can be important for improving the memory usage in transformer-based diffusion models, particularly for high-resolution images.

            ## Additional pipeline parameters ##
            scheduler: Manages the denoising process during diffusion. You can use schedulers like `FlowMatchEulerDiscreteScheduler` to handle different noise prediction strategies and timestep management.
            text_encoder: If you're using a text-to-image diffusion model, you'll need a text encoder (e.g., CLIP or a similar transformer-based encoder) to convert text prompts into embeddings. Generally encoder_2 is `openai/clip-vit-large-patch14`
            text_encoder_2: If you're using a text-to-image diffusion model, you'll need a text encoder (e.g., CLIP or a similar transformer-based encoder) to convert text prompts into embeddings. Generally encoder_2 is `google/t5-v1_1-xxl`
            tokenizer: Required for converting text inputs into token IDs, used when working with text-to-image diffusion models.
            tokenizer_2: Required for converting text inputs into token IDs, used when working with text-to-image diffusion models.

            ## return intermediate results or logits ## 
            intermediate_layer_outputs: Parameters specifically required for flow-matching, such as the noise schedule, matching loss functions, and intermediate layer outputs, critical for models like FLUX.

            ## quantization and sparsification ##
            dtype: Defines the precision of the model weights and activations. For diffusion, you can use float32, float16, or bfloat16, based on hardware availability.
            quantization: Optional but useful for optimizing model memory and speed. Formats like NF4 or 8-bit quantization can be applied to diffusion models to reduce resource usage.
            cpu_offload_gb: Amount of CPU memory used for offloading model weights during training or inference, enabling larger models to be run on limited GPU resources.
            swap_space: For managing the model's memory when using large configurations, especially when running multiple parallel requests.

            ## GPU & Parallelism Configuration ##
            tensor_parallel_size: Specifies the number of GPUs to use for distributed execution. In diffusion models, this can be used to split the workload across multiple GPUs.
            gpu_memory_utilization: A critical parameter in diffusion models, determining how much GPU memory to reserve for model weights and activations.
            pipeline_parallel_size: Used for pipeline parallelism in large diffusion models to reduce memory load by distributing layers across devices.
            disable_custom_all_reduce: Used in parallel execution scenarios to control the behavior of all-reduce operations.

            ## Inference parameters ##
            sampling_steps: Number of timesteps to sample from the diffusion process, balancing quality and efficiency.
            scheduler_type: Determines the noise schedule, impacting how noise is introduced and removed across diffusion steps. E.g., DDIM, PNDM, or more custom schedulers.
            seed: Sets the seed for random number generation, useful for ensuring reproducibility in stochastic processes like diffusion.

            ## Scheduler parameters ## 
            scheduler_steps: Controls the number of iterations the scheduler will take to diffuse the data back into the image space.
            scheduler_config: Contains additional configuration for the diffusion steps such as the noise type, step size, and step interpolation methods (e.g., linear vs. quadratic schedules).

    """
    def __init__(
        self,
        model: str,
        lora_adapters: Optional[str] = None,
        controlnet_model: Optional[str] = None,
        use_memory_efficient_attention: bool = False,
        scheduler: Optional[str] = None,
        text_encoder: Optional[str] = None,
        text_encoder_2: Optional[str] = None,
        tokenizer: Optional[str] = None,
        tokenizer_2: Optional[str] = None,
        intermediate_layer_outputs: bool = False,
        dtype: Optional[str] = "fp16",
        quantization: Optional[str] = None,
        cpu_offload_gb: Optional[int] = 0,
        swap_space: Optional[str] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: Optional[str] = None,
        pipeline_parallel_size: Optional[int] = None,
        disable_custom_all_reduce: bool = False,
        sampling_steps: Optional[int] = 50,
        scheduler_type: Optional[str] = None,
        seed: int = 1,
        scheduler_steps: int = 1000,
        scheduler_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        self.engine_args = ContinuityArgsParser(
            model=model,
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            dtype=dtype,
            quantization=quantization,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            sampling_steps=sampling_steps,
            scheduler_type=scheduler_type,
            scheduler_steps=scheduler_steps,
            scheduler_config=scheduler_config or {},
            kwargs=kwargs
        )

        self.diffusion_engine = DiffusionEngine.from_engine_args(
            self.engine_args, usage_context="TBD"
        )
        self.request_counter = Counter()

    def get_tokenizer(self) -> Any:
        """Retrieve the tokenizer from the diffusion engine."""
        return self.diffusion_engine.get_tokenizer_group(TokenizerGroup).tokenizer

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Set a tokenizer, with caching logic if required."""
        group = self.diffusion_engine.get_tokenizer_group(TokenizerGroup)
        if tokenizer.__class__.__name__.startswith("Cached"):
            group.tokenizer = tokenizer
        else:
            group.tokenizer = get_cached_tokenizer(tokenizer)

    def generate(
        self,
        prompt: Union[str, Sequence[str]],
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        priority: Optional[List[int]] = None,
    ) -> List[Any]:
        """
        Generate images from given prompts using the diffusion model.

        Args:
            prompt: Input prompts or batch of prompts.
            prompt_token_ids: Optional token IDs corresponding to the prompts.
            use_tqdm: Whether to display a progress bar.
            priority: Optional list of priorities for batch requests.

        Returns:
            A list of generated outputs.
        """
        outputs = self._run_engine(use_tqdm=use_tqdm)
        return DiffusionEngine.validate_outputs(outputs, RequestOutput)

    def _run_engine(self, *, use_tqdm: bool) -> List[Union[Any, EmbeddingRequestOutput]]:
        """Internal method to execute the engine logic."""
        num_requests = self.diffusion_engine.do_something()
        outputs = [RequestOutput(request_id=i) for i in range(num_requests)]
        if use_tqdm:
            outputs = list(tqdm(outputs, desc="Generating outputs"))
        return sorted(outputs, key=lambda x: int(x.request_id))

@dataclass
class RequestOutput:
    """Represents the output of a single generation request."""
    request_id: int

def get_cached_tokenizer(tokenizer: Any) -> Any:
    # Logic for retrieving or caching tokenizers
    pass

