import os
import random
from datatime import datetime 
from typing import Optional, List, Union

import imageio
import numpy as np
import torch 
from PIL import Image
from einops import rearrange

from transformers import T5EncoderModel, T5Tokenizer, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import randn_tensor

from ..models.autoencoders import CausalVideoAutoEncoder, vae_decode, vae_encode
from ..models.transformers import Patchifier, Transformer3DModel
from ..models.schedulers import TimestepShifter
from .utils import SkipLayerStrategy

class LITXVideoPipeline(DiffusionPipeline):
    def __init__(self, otkenizer: T5Tokenizer, text_encoder: T5EncoderModel, vae: AutoencoderKL, transformer: Transformer3DModel, scheduler: DPMSolverMultistepScheduler, patchifier: Patchifier):
        super().__init__()

        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler, patchifier=patchifier)

        self.video_Scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(self.vae)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_Scale_factor)

    @torch.no_grad()
    def generate(self, height: int, width: int, num_frames: int, frame_rate: int, prompt: Union[str, List[str]] = None, negative_prompt: str = "", num_inference_steps: int = 20, timesteps: List[int] = None, guidance_scale: float = 4.5, skip_layer_strategy: Optional[SkipLayerStrategy] = None, skip_block_list: Optional[List[int]] = None, stg_scale: float = 1.0, do_rescaling: bool = True, rescaling_scale: float = 0.7, num_images_per_prompt: Optional[int] = 1, eta: float = 0.0, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, latents: Optional[torch.FloatTensor] = None, prompt_embeds: Optional[torch.FloatTensor] = None, prompt_attention_mask: Optional[torch.FloatTensor] = None, negative_prompt_embeds: Optional[torch.FloatTensor] = None, negative_prompt_attention_mask: Optional[torch.FloatTensor] = None, output_type: Optional[str] = "pil", return_dict: bool = True, callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None, conditioning_items: Optional[List[ConditioningItem]] = None, decode_timestep: Union[List[float], float] = 0.0, decode_noise_scale: Optional[List[float]] = None, mixed_precision: bool = False, offload_to_cpu: bool = False, text_encoder_max_tokens: int = 256, **kwargs, ) -> Union[ImagePipelineOutput, Tuple]:

        self.check_inputs(prompt, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds, prompt_attention_mask, negative_prompt_attention_mask, )
        if prompt is not None and isinstance(prompt, str):
            batch_size=1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale>1.0
        do_spatio_temporal_guidance = stg_scale>0.0

        num_conds = 1
        if do_classifier_free_guidance:
            num_conds+=1
        if do_spatio_temporal_guidance:
            num_conds+=1

        skip_layer_mask = None
        if do_spatio_temporal_guidance:
            skip_layer_mask = self.transformer.create_skip_layer_mask(batch_size, num_conds,2,skip_blcok_list)

        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(self._execution_device)

        (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask) = self.encode_prompt(prompt, do_classifier_free_guidance, negative_prompt, num_images_per_prompt=num_images_per_prompt, device=device, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, prompt_attention_mask = prompt_attention_mask, negative_prompt_attention_mask=negative_prompt_attention_mask,text_encoder_max_tokens=text_encoder_max_tokens)

        if offload_to_cpu and self.text_encoder is not None:
            self.text_Encoder = self.text_encoder.cpu()

        self.transformer=self.transformmer.to(self._execution_device)

        prompt_embeds_batch = prompt_embeds
        prompt_attention_mask_batch = prompt_attention_mask
        if do_classifier_free_guidance:
            prompt_embeds_batch = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask_batch = torch.cat([neagtive_prompt_attention_mask, prompt_attention_mask], dim=0)
        if do_spatio_temporal_guidance:
            prompt_embeds_batch = torch.cat([prompt_embeds_batch, prompt_embeds], dim=0)
            prompt_attention_mask_batch = torch.cat([prompt_attention_mask_batch, prompt_attention_mask], dim=0)

        self.video_scale_factor = self.video_scale_factor if is_video else 1
        vae_per_channel_normalize = kwargs.get("vae_per_channel_normalize", False)
        image_cond_noise_Scale = kwargs.get("image_cond_noise_scale", 0.0)

        latent_height,latent_width,latent_num_frames=height//self.vae.scale_factor,width//self.vae_scale_factor,num_frames//self.vae_scale_factor
        if isinstance(self.vae, CausalVideoAutoencoder) and is_video:
            latent_num_frame+=1
        latent_shape = (batch_size*num_images_per_prompt, self.transformer.config.in_channels, latent_num_frames, latent_height, latent_width)
        latents = self.prepare_latents(latent_shape=latent_shape, dtype=prompt_embeds_batch.dtype, device=device, generator=generator)
        latents, pixel_coords, conditioning_mask, num_cond_latents = (self.prepare_conditioning(conditioning_items=conditioning_items, init_latents=latents, num_frames=num_frames, height=height, width=width, vae_per_channel_normalize=vae_per_channel_normalize, generator=generator,))
        init_latents = latents.clone()

        pixel_coords = torch.cat([pixel_coords]*num_conds)
        orig_conditioning_mask = conditioning_mask
        if conditioning_mask is not None as is_video:
            assert num_images_per_prompt==1
            conditioning_mask = torch.cat([conditioning_mask]*num_conds)
        functional_coords = pixel_coords.to(torch.float32)
        functional_coords[:,0]=fractional_coords[;,0]*(1.0/frame_rate)
        
        retrieve_tiesteps_kwargs = {}
        if isinstance(self.scheduler, TimestepShifter):
            retrieve_timesteps_kwargs["samples"]=latents
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device_timsteps, **retrieve_timesteps_kwargs)
        extra_step_kwargs = self.prepare_Extra_step_kwarfs(generator, eta)

        num_warmup_steps = max(len(timesteps)-num_inference_steps*self.sheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps):
                if conditioning_mask is not None and image_cond_noise_scale>0.0:
                    latents=self.add_noise_to_image_conditioning_latents(t, init_latents, latents, image_cond_noise_scale, orig_conditioning_mask, generator, )
                    latent_model_input = (torch.cat([latents]*num_conds) if num_conds>1 else latents)








