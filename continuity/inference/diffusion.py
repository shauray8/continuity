import itertools
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union, cast, overload

from tqdm import tqdm

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
        text_encoder: Optiional[str] = None,
        text_encoder_2: Optiional[str] = None,
        tokenizer: Optiional[str] = None,
        tokenizer_2: Optiional[str] = None,
        intermediate_layer_outputs: bool = False,
        dtype: Optional[str] = "fp16",
        quantization: Optional[str] = None,
        cpu_offload_gb: Optional[int] = 0
        swap_space: Optional[str] = None,
        tensor_parallel_size: Optional[int] = 1,
        gpu_memory_utilization: Optional[str] = None,
        pipeline_parallel_size: Optional[str] = None,
        disable_custom_all_reduce: Optional[bool] = False,
        sampling_steps: Optional[str] = None, 
        scheduler_type: Optinal[str] = None,
        seed: Optional[int] = 1,
        scheduler_steps: Optional[int] = 1000,
        scheduler_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:

        engine_args = ContinuityArgsParser(
            model=model,
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer=tokenizer_2,
            dtype=dtype,
            quantization=quantization,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            sampling_steps=sampling_steps,
            scheduler_type=scheduler_type,
            scheduler_steps=scheduler_steps,
            **kwargs
        )

        self.diffusion_engine = DiffusionEngine.from_engine_args(
            engine_args, usage_context=UsageContext.TBD)

        self.request_counter = Counter()

    ## add similar stuff for text encoder
    def get_tokenizer(self) -> AnyTokenizer:
        return self.diffusion_engine.get_tokenizer_group(TokenizerGroup).tokenizer

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        tokenizer_group = self.diffusion_engine.get_tokenizer_group(TokenizerGroup)

        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            tokenizer_group.tokenizer = tokenizer
        else:
            tokenizer_group.tokenizer = get_cached_tokenizer(tokenizer)

    def generate(
        self,
        prompt: Union[Union[PromptType, SequenceType]],
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]],
        use_tqdm: bool = True,
        prioroty: Optional[List[int]] = None,
    ) -> List[RequestOutput]:

        """
        Generates the outputs for the input conditions using a diffusion model.
        This method automatically batches the given conditions (e.g., text prompts, image conditions), considering memory constraints. For the best performance, input all your conditions into a single list and pass it to this method.

        Args:
            prompt: 
                The prompts to the Diffusion. You may pass a sequence of prompts for batch inference.
            num_inference_steps: The number of diffusion steps to run for inference. More steps typically yield higher quality outputs but require more computation.
            guidance_scale: 
                For guided diffusion models (e.g., classifier-free guidance), this controls the trade-off between fidelity to the condition (prompt) and the diversity of the output.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            seed: 
                The seed for the random number generator, ensuring reproducibility of the diffusion process.
            use_tqdm: 
                Whether to use a progress bar (via tqdm) to display the status of the diffusion process.
            priority: 
                The priority of the requests, which may be useful in cases where multiple generations are queued. If a priority scheduling policy is enabled, this parameter can ensure that certain requests are prioritized over others.

        Returns:
            A list of ``RequestOutput`` objects containing the
            generated completions in the same order as the input prompts.

        """

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return DiffusionEngine.validate_outputs(outputs, RequestOutput)

    def _run_engine(
        self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        if use_tqdm:
            num_requests = self.diffusion_engine.do_something()

        return sorted(outputs, key=lambda x: int(x.request_id))
