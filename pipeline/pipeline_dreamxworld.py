import inspect
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from pathlib import Path
from transformers import T5Tokenizer

from models import (AutoencoderKLWan, AutoTokenizer,
                    Wan2_2Transformer3DModel, WanT5EncoderModel)
from utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                              get_sampling_sigmas)
from utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.utils import get_image_to_video_latent, get_image_latent, calculate_padding

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        pass
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


@dataclass
class WanPipelineOutput(BaseOutput):
    r"""
    Output class for CogVideo pipelines.

    Args:
        video (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
            `(batch_size, num_frames, channels, height, width)`.
    """

    videos: torch.Tensor


class Wan2_2_CameraPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    _optional_components = ["transformer_2"]
    model_cpu_offload_seq = "text_encoder->transformer_2->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: Wan2_2Transformer3DModel,
        transformer_2: Wan2_2Transformer3DModel = None,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, 
            transformer_2=transformer_2, scheduler=scheduler
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cache_prompt_embeds: bool = False,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        def _cached_t5(prompt_value, cache_name):
            prompt_key = tuple(prompt_value) if isinstance(prompt_value, list) else (prompt_value,)
            key = (
                prompt_key,
                int(num_videos_per_prompt),
                int(max_sequence_length),
                str(device),
                str(dtype or self.text_encoder.dtype),
            )
            cache = getattr(self, cache_name, None)
            if cache is None:
                cache = {}
                setattr(self, cache_name, cache)
            if key not in cache:
                cache[key] = self._get_t5_prompt_embeds(
                    prompt=prompt_value,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )
                if len(cache) > 8:
                    cache.pop(next(iter(cache)))
            return cache[key]

        if prompt_embeds is None:
            if cache_prompt_embeds:
                prompt_embeds = _cached_t5(prompt, "_positive_prompt_embed_cache")
            else:
                prompt_embeds = self._get_t5_prompt_embeds(
                    prompt=prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            if cache_prompt_embeds:
                negative_prompt_embeds = _cached_t5(negative_prompt, "_negative_prompt_embed_cache")
            else:
                negative_prompt_embeds = self._get_t5_prompt_embeds(
                    prompt=negative_prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae.temporal_compression_ratio + 1,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance, noise_aug_strength
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if mask is not None:
            mask = mask.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask = []
            for i in range(0, mask.shape[0], bs):
                mask_bs = mask[i : i + bs]
                mask_bs = self.vae.encode(mask_bs)[0]
                mask_bs = mask_bs.mode()
                new_mask.append(mask_bs)
            mask = torch.cat(new_mask, dim = 0)
            # mask = mask * self.vae.config.scaling_factor

        if masked_image is not None:
            masked_image = masked_image.to(device=device, dtype=self.vae.dtype)
            bs = 1
            new_mask_pixel_values = []
            for i in range(0, masked_image.shape[0], bs):
                mask_pixel_values_bs = masked_image[i : i + bs]
                mask_pixel_values_bs = self.vae.encode(mask_pixel_values_bs)[0]
                mask_pixel_values_bs = mask_pixel_values_bs.mode()
                new_mask_pixel_values.append(mask_pixel_values_bs)
            masked_image_latents = torch.cat(new_mask_pixel_values, dim = 0)
            # masked_image_latents = masked_image_latents * self.vae.config.scaling_factor
        else:
            masked_image_latents = None

        return mask, masked_image_latents

    def prepare_control_latents(
        self, control, control_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the control to latents shape as we concatenate the control to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision

        if control is not None:
            control = control.to(device=device, dtype=dtype)
            bs = 1
            new_control = []
            for i in range(0, control.shape[0], bs):
                control_bs = control[i : i + bs]
                control_bs = self.vae.encode(control_bs)[0]
                control_bs = control_bs.mode()
                new_control.append(control_bs)
            control = torch.cat(new_control, dim = 0)

        if control_image is not None:
            control_image = control_image.to(device=device, dtype=dtype)
            bs = 1
            new_control_pixel_values = []
            for i in range(0, control_image.shape[0], bs):
                control_pixel_values_bs = control_image[i : i + bs]
                control_pixel_values_bs = self.vae.encode(control_pixel_values_bs)[0]
                control_pixel_values_bs = control_pixel_values_bs.mode()
                new_control_pixel_values.append(control_pixel_values_bs)
            control_image_latents = torch.cat(new_control_pixel_values, dim = 0)
        else:
            control_image_latents = None

        return control, control_image_latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        video: Union[torch.FloatTensor] = None,
        control_camera_video: Union[torch.FloatTensor] = None,
        start_image: Union[torch.FloatTensor] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        boundary: float = 0.875,
        comfyui_progressbar: bool = False,
        shift: int = 5,
        start_image_lock_frames: int = 1,
        cache_prompt_embeds: bool = False,
        semantic_layout_labels: Optional[torch.Tensor] = None,
        semantic_layout_features: Optional[torch.Tensor] = None,
        semantic_layout_progress: Optional[torch.Tensor] = None,
        map_strength: float = 0.0,
        # R1b world warm start: cached-world latent prior + coverage mask for
        # this segment (built by utils/world_warm_start from STRICTLY EARLIER
        # segments' rendered frames), plus the denoise step index s0 to start
        # from. s0 is chosen by sigma (shift=3: s0=15 -> sigma=0.50, 5 steps),
        # NOT by step count — a naive half schedule (s0=10, sigma=0.75)
        # destroys 75% of the prior.
        warm_start_latents: Optional[torch.FloatTensor] = None,
        warm_start_mask: Optional[torch.FloatTensor] = None,
        warm_start_step_index: int = 0,
        warm_start_info: Optional[Dict[str, Any]] = None,
        # anchor10: per-step hard re-lock of covered pixels onto the prior's
        # flow-matching trajectory. Generalizes the frame-0 start-image lock
        # (measured perfect: 1022.7 vs bank 1018.4) to ALL covered pixels at
        # EVERY denoise step. Knee experiment (warm10/cold10) showed the prior
        # is a RESCUER not a polluter (warm10 >> cold10 2.5-65x per fresh-prior
        # seg); the kill mechanism is warm-from-warm recursion + short-schedule
        # drift, not the prior itself. Anchoring nails covered pixels so the
        # short schedule cannot drift them.
        world_warm_anchor: bool = False,
    ) -> Union[WanPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        Args:

        Examples:

        Returns:

        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        weight_dtype = self.text_encoder.dtype

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            cache_prompt_embeds=cache_prompt_embeds,
        )
        if do_classifier_free_guidance:
            in_prompt_embeds = negative_prompt_embeds + prompt_embeds
        else:
            in_prompt_embeds = prompt_embeds

        # 4. Prepare timesteps
        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, mu=1)
        elif isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, FlowDPMSolverMultistepScheduler):
            sampling_sigmas = get_sampling_sigmas(num_inference_steps, shift)
            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        
        if comfyui_progressbar:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(num_inference_steps + 2)

        # 5. Prepare latents.
        if video is not None:
            video_length = video.shape[2]
            init_video = self.image_processor.preprocess(rearrange(video, "b c f h w -> (b f) c h w"), height=height, width=width) 
            init_video = init_video.to(dtype=torch.float32)
            init_video = rearrange(init_video, "(b f) c h w -> b c f h w", f=video_length)
        else:
            init_video = None

        latent_channels = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            weight_dtype,
            device,
            generator,
            latents,
        )
        if comfyui_progressbar:
            pbar.update(1)


        # Prepare camera control latent variables
        if control_camera_video is not None and isinstance(control_camera_video, torch.Tensor):
            # Rearrange dimensions
            # Concatenate and transpose dimensions
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                    control_camera_video[:, :, 1:]
                ], dim=2
            ).transpose(1, 2)

            # Reshape, transpose, and view into desired shape
            b, f, c, h, w = control_camera_latents.shape
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
            control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)

        elif control_camera_video is not None and isinstance(control_camera_video, dict):
            control_camera_latents = {}
            for key, value in control_camera_video.items():
                control_camera_latents[key] = control_camera_video[key].unsqueeze(0)


        mask = torch.zeros(1, 1, latents.size()[2], latents.size()[3], latents.size()[4]).to(device, weight_dtype)

        # === R1b world warm start: cached-world SDEdit init + step skipping ===
        # Replace the pure-noise init with a noise-consistent blend at
        # sigma(t_{s0}) and run only the tail s0..N-1 of the schedule. The
        # scheduler's step() cold-starts by timestep VALUE (first-order
        # update), so slicing the local `timesteps` is sufficient. Placed
        # BEFORE the start-image block so the frame-0 lock below applies on
        # top of the warm blend. The blend keeps ONE noise level for the whole
        # frame — covered pixels carry the cached-world prior, uncovered
        # pixels a zero (grey) x0 prior — so the network never sees spatially
        # varying noise (the failure mode that killed E2-style refresh skip).
        _warm_s0 = 0
        _warm_seg = int(warm_start_info.get("seg_idx", -1)) if warm_start_info else -1
        # anchor10 state: saved for per-step re-lock inside the denoise loop.
        _anchor_warm = None
        _anchor_wmask = None
        _anchor_eps = None
        _anchor_active = bool(world_warm_anchor)
        _anchor_release = float(getattr(self, "_warm_anchor_release", 0.0))
        if (warm_start_latents is not None and int(warm_start_step_index) > 0
                and len(timesteps) > 1):
            warm = warm_start_latents.to(device=device, dtype=latents.dtype)
            wmask = warm_start_mask.to(device=device, dtype=latents.dtype)
            if (tuple(warm.shape) == tuple(latents.shape)
                    and tuple(wmask.shape) == (latents.shape[0], 1, *latents.shape[2:])):
                _warm_s0 = int(min(int(warm_start_step_index), len(timesteps) - 1))
                _sigma = float(self.scheduler.sigmas[_warm_s0].item())
                wmask = wmask.clone()
                wmask[:, :, 0] = 0.0  # frame 0 stays locked to the start image below
                eps = randn_tensor(latents.shape, generator=generator,
                                   device=device, dtype=latents.dtype)
                latents = (1.0 - _sigma) * (wmask * warm) + _sigma * eps
                # Save for per-step anchoring: the prior latent and the SAME
                # per-segment fixed noise (so the anchor trajectory is a single
                # deterministic FM path, not re-randomized each step).
                if _anchor_active:
                    _anchor_warm = warm
                    _anchor_wmask = wmask
                    _anchor_eps = eps
                timesteps = timesteps[_warm_s0:]
                _cov = float(warm_start_info.get("coverage", float("nan"))) if warm_start_info else float("nan")
                _anchor_tag = " +anchor" if _anchor_active else ""
                print(f"[world_warm_start] seg={_warm_seg} s0={_warm_s0} sigma={_sigma:.3f} "
                      f"steps_run={len(timesteps)} coverage={_cov:.3f}{_anchor_tag}", flush=True)
            else:
                print(f"[world_warm_start] seg={_warm_seg} shape mismatch "
                      f"warm={tuple(warm.shape)} mask={tuple(wmask.shape)} "
                      f"latents={tuple(latents.shape)}; running cold", flush=True)

        if start_image is not None:
            # I2V
            if isinstance(start_image, (str, Path)): 
                start_image, _, _ = get_image_to_video_latent(start_image, None, video_length=num_frames, sample_size=[height, width])
               

            video_length = start_image.shape[2]
            start_image = self.image_processor.preprocess(rearrange(start_image, "b c f h w -> (b f) c h w"), height=height, width=width) 
            start_image = start_image.to(dtype=torch.float32)
            start_image = rearrange(start_image, "(b f) c h w -> b c f h w", f=video_length)
            
            start_image_latentes = self.prepare_control_latents(
                None,
                start_image,
                batch_size,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance
            )[1]

            if self.vae.spatial_compression_ratio >= 16:
                # start_image_lock_frames: lock the first K latent frames to the start_image
                # latent with linearly decaying strength (frame 0 = full lock, frame K-1 ~ 0).
                # This suppresses cross-segment roadside-building jumps by giving the model a
                # smooth handoff zone instead of a hard 1-frame lock followed by free generation.
                num_latent_frames = latents.shape[2]
                lock_k = max(1, min(int(start_image_lock_frames), num_latent_frames))
                if lock_k <= 1:
                    mask[:, :, 1:] = 1
                else:
                    # decay from 0 (frame 0, full lock) to 1 (frame lock_k-1, full free)
                    decay = torch.linspace(0.0, 1.0, lock_k, device=mask.device, dtype=mask.dtype)
                    mask[:, :, :lock_k] = decay.view(1, 1, lock_k, 1, 1)
                    mask[:, :, lock_k:] = 1
                latents = (1 - mask) * start_image_latentes + mask * latents

            elif self.vae.spatial_compression_ratio < 16:
                bsz, _, num_latent_frames, latent_height, latent_width = latents.shape # [b, c, f, h, w]

                mask_latents = torch.ones((bsz, 1, num_frames, latent_height, latent_width)).to(device, weight_dtype)
                mask_latents[:, :, 1:] = 0
                mask_latents = torch.concat(
                    [
                        torch.repeat_interleave(mask_latents[:, :, 0:1], repeats=4, dim=2), 
                        mask_latents[:, :, 1:]
                    ], dim=2
                ) # torch.Size([1, 1, 84, h, w])
                mask_latents = mask_latents.view(mask_latents.shape[0], mask_latents.shape[2] // 4, 4, mask_latents.shape[3], mask_latents.shape[4]) # 1, 21, 4, h, w
                mask_latents = mask_latents.transpose(1, 2) # 1, 4, 21, h, w
                
                
                masked_video = torch.cat([start_image[:, :, 0:1], start_image.new_zeros(start_image.shape[0], start_image.shape[1], num_frames-1, start_image.shape[3], start_image.shape[4])], dim=2)
                _, masked_video_latents = self.prepare_mask_latents(
                    None,
                    masked_video,
                    batch_size,
                    height,
                    width,
                    weight_dtype,
                    device,
                    generator,
                    do_classifier_free_guidance,
                    noise_aug_strength=None,
                )



        if comfyui_progressbar:
            pbar.update(1)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        target_shape = (self.vae.latent_channels, (num_frames - 1) // self.vae.temporal_compression_ratio + 1, height // self.vae.spatial_compression_ratio, width // self.vae.spatial_compression_ratio)
        seq_len = math.ceil((target_shape[2] * target_shape[3]) / (self.transformer.config.patch_size[1] * self.transformer.config.patch_size[2]) * target_shape[1])

        layout_labels_input = None
        layout_progress_input = None
        strength = float(map_strength)
        if not math.isfinite(strength):
            raise ValueError("map_strength must be finite")
        if semantic_layout_labels is not None and semantic_layout_features is not None:
            raise ValueError("provide semantic_layout_labels or semantic_layout_features, not both")
        if semantic_layout_labels is not None or semantic_layout_features is not None:
            if strength <= 0:
                raise ValueError("semantic layout conditioning requires map_strength > 0")
            labels = torch.as_tensor(
                semantic_layout_features if semantic_layout_features is not None else semantic_layout_labels,
                device=device,
            )
            expected_ndim = 5 if semantic_layout_features is not None else 4
            if labels.ndim != expected_ndim:
                name = "semantic_layout_features" if semantic_layout_features is not None else "semantic_layout_labels"
                raise ValueError(f"{name} must have {expected_ndim} dimensions, got {tuple(labels.shape)}")
            if semantic_layout_features is not None and not labels.dtype.is_floating_point:
                raise TypeError("semantic_layout_features must use a floating dtype")
            if semantic_layout_features is None and (labels.dtype.is_floating_point or labels.dtype == torch.bool):
                raise TypeError("semantic_layout_labels must use an integer dtype")
            if int(labels.shape[0]) not in {1, batch_size}:
                raise ValueError("semantic_layout_labels batch is incompatible with the prompt batch")
            if int(labels.shape[1]) != int(target_shape[1]):
                raise ValueError(
                    "semantic layout frame count must match the VAE latent frame count: "
                    f"{labels.shape[1]} != {target_shape[1]}"
                )
            if int(labels.shape[0]) == 1 and batch_size > 1:
                labels = labels.expand(batch_size, *([-1] * (labels.ndim - 1)))
            progress = torch.as_tensor(
                semantic_layout_progress
                if semantic_layout_progress is not None
                else torch.linspace(0, 1, labels.shape[1]),
                device=device,
                dtype=torch.float32,
            )
            if progress.ndim == 1:
                progress = progress.unsqueeze(0)
            if tuple(progress.shape) == (1, labels.shape[1]) and labels.shape[0] > 1:
                progress = progress.expand(labels.shape[0], -1)
            if tuple(progress.shape) != (labels.shape[0], labels.shape[1]):
                raise ValueError("semantic_layout_progress must have shape [B,F] or [F]")
            if not bool(torch.isfinite(progress).all()):
                raise ValueError("semantic_layout_progress must contain only finite values")
            if bool(((progress < 0) | (progress > 1)).any()):
                raise ValueError("semantic_layout_progress values must lie in [0, 1]")
            layout_labels_input = torch.cat([labels, labels], dim=0) if do_classifier_free_guidance else labels
            layout_progress_input = torch.cat([progress, progress], dim=0) if do_classifier_free_guidance else progress
        elif strength != 0:
            raise ValueError("map_strength is nonzero but no semantic layout was provided")

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self.transformer.num_inference_steps = num_inference_steps
        _denoise_t0 = time.time()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # warm-started segments run the sliced tail; report the TRUE
                # schedule position (mapctrl gamma(t) reads current_steps).
                self.transformer.current_steps = i + _warm_s0

                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prepare control latent variables
                if control_camera_video is not None and isinstance(control_camera_video, torch.Tensor):
                    control_camera_latents_input = (
                        torch.cat([control_camera_latents] * 2) if do_classifier_free_guidance else control_camera_latents
                    ).to(device, weight_dtype)
                elif control_camera_video is not None and isinstance(control_camera_video, dict):
                    control_camera_latents_input = {}
                    for key, value in control_camera_latents.items():
                        control_camera_latents_input[key] = (
                            torch.cat([value] * 2) if do_classifier_free_guidance else value
                        ).to(device, weight_dtype)
                
                # for wan2.2-a14b
                y = None
                if self.vae.spatial_compression_ratio < 16 and mask_latents is not None:
                    mask_input = torch.cat([mask_latents] * 2) if do_classifier_free_guidance else mask_latents
                    masked_video_latents_input = (
                        torch.cat([masked_video_latents] * 2) if do_classifier_free_guidance else masked_video_latents
                    )
                    y = torch.cat([mask_input, masked_video_latents_input], dim=1).to(device, weight_dtype) 
              

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                if self.vae.spatial_compression_ratio >= 16:
                    temp_ts = ((mask[0][0][:, ::2, ::2]) * t).flatten()
                    temp_ts = torch.cat([
                        temp_ts,
                        temp_ts.new_ones(seq_len - temp_ts.size(0)) * t
                    ])
                    temp_ts = temp_ts.unsqueeze(0)
                    timestep = temp_ts.expand(latent_model_input.shape[0], temp_ts.size(1))
                else:
                    timestep = t.expand(latent_model_input.shape[0])
                
                # transformer_2 is high 
                if self.transformer_2 is not None:
                    if t >= boundary * self.scheduler.config.num_train_timesteps:
                        local_transformer = self.transformer_2
                    else:
                        local_transformer = self.transformer
                       
                else:
                    local_transformer = self.transformer
                
                # === SMG: compute per-step spatial map guidance ===
                _smg_signal = None
                _smg_cfg = getattr(self, "_smg_config", None)
                if _smg_cfg is not None:
                    try:
                        from map_guidance import prepare_map_guidance
                        # grid_sizes: derive from latent_model_input shape
                        # latent_model_input: [B, C, F, H, W] after patch_embedding
                        # But we need grid_sizes BEFORE patch_embedding
                        # Use the pre-patchify shape from seq_len
                        _B = latent_model_input.shape[0]
                        _F = latent_model_input.shape[2]
                        _H = latent_model_input.shape[3]
                        _W = latent_model_input.shape[4]
                        # Grid sizes after patch_embedding (patch=(1,2,2))
                        _grid_h = _H // 2
                        _grid_w = _W // 2
                        _grid_sizes = torch.tensor([[_F, _grid_h, _grid_w]] * _B, dtype=torch.long)
                        _smg_signal = prepare_map_guidance(
                            map_path=_smg_cfg['map_path'],
                            grid_sizes=_grid_sizes,
                            seq_len=seq_len,
                            device=latent_model_input.device,
                            dtype=weight_dtype,
                            inject_layers=_smg_cfg['inject_layers'],
                            max_strength=_smg_cfg['max_strength'],
                        )
                    except Exception as e:
                        print(f"  [SMG] Warning: failed to compute guidance: {e}")
                        _smg_signal = None

                # === SML: compute semantic map latent ===
                _sml_signal = None
                _sml_cfg = getattr(self, "_sml_config", None)
                if _sml_cfg is not None:
                    try:
                        from map_latent import prepare_map_latent
                        _B = latent_model_input.shape[0]
                        _F = latent_model_input.shape[2]
                        _H = latent_model_input.shape[3]
                        _W = latent_model_input.shape[4]
                        _grid_h = _H // 2
                        _grid_w = _W // 2
                        _grid_sizes = torch.tensor([[_F, _grid_h, _grid_w]] * _B, dtype=torch.long)
                        _sml_signal = prepare_map_latent(
                            map_path=_sml_cfg["map_path"],
                            grid_sizes=_grid_sizes,
                            seq_len=seq_len,
                            device=latent_model_input.device,
                            dtype=weight_dtype,
                            inject_layers=_sml_cfg["inject_layers"],
                            max_strength=_sml_cfg["max_strength"],
                            dim=3072,
                        )
                    except Exception as e:
                        print(f"  [SML] Warning: failed to compute latent: {e}")
                        _sml_signal = None

                # === MapState adapter (dense_residual): trained residual ===
                # Produces the SAME list-of-tensors format as SML and flows
                # through the SAME map_latent kwarg. Takes precedence over SML
                # when both are configured (mutual exclusivity enforced at setup).
                _mapstate_signal = None
                _ms_cfg = getattr(self, "_mapstate_config", None)
                if _ms_cfg is not None:
                    try:
                        from mapstate import prepare_mapstate_signal
                        _B = latent_model_input.shape[0]
                        _F = latent_model_input.shape[2]
                        _H = latent_model_input.shape[3]
                        _W = latent_model_input.shape[4]
                        _grid_h = _H // 2
                        _grid_w = _W // 2
                        _grid_sizes = torch.tensor([[_F, _grid_h, _grid_w]] * _B, dtype=torch.long)
                        _ms_route = _ms_cfg.get("route_polyline")
                        _ms_prog = getattr(self, "_mapstate_route_progress", (0.0, 1.0))
                        _mapstate_signal = prepare_mapstate_signal(
                            map_path=_ms_cfg["map_path"],
                            grid_sizes=_grid_sizes,
                            seq_len=seq_len,
                            route_polyline=_ms_route,
                            route_progress=_ms_prog,
                            inject_blocks=(12,),
                            dim=3072,
                            crop_extent_px=_ms_cfg.get("crop_extent_px", 128.0),
                            strength=_ms_cfg.get("strength", 1.0),
                            ckpt=_ms_cfg["ckpt"],
                            representation=_ms_cfg.get("representation", "s-f"),
                            device="cpu",
                            dtype=weight_dtype,
                        )
                    except Exception as e:
                        print(f"  [MapState] Warning: failed to compute signal: {e}")
                        import traceback; traceback.print_exc()
                        _mapstate_signal = None
                # MapState takes precedence over SML for the map_latent kwarg.
                _map_latent_signal = _mapstate_signal if _mapstate_signal is not None else _sml_signal

                # === MKI: compute map key/value for map-aware attention ===
                _mki_keys = None
                _mki_values = None
                _mki_cfg = getattr(self, "_mki_config", None)
                if _mki_cfg is not None:
                    try:
                        from map_key_injection import prepare_map_key_value
                        _B = latent_model_input.shape[0]
                        _F = latent_model_input.shape[2]
                        _H = latent_model_input.shape[3]
                        _W = latent_model_input.shape[4]
                        _grid_h = _H // 2
                        _grid_w = _W // 2
                        _grid_sizes = torch.tensor([[_F, _grid_h, _grid_w]] * _B, dtype=torch.long)
                        _mki_keys, _mki_values = prepare_map_key_value(
                            map_path=_mki_cfg["map_path"],
                            grid_sizes=_grid_sizes,
                            seq_len=seq_len,
                            device=latent_model_input.device,
                            dtype=weight_dtype,
                            inject_layers=_mki_cfg["inject_layers"],
                            key_strength=_mki_cfg["key_strength"],
                            value_strength=_mki_cfg["value_strength"],
                            dim=3072,
                            num_heads=24,
                        )
                    except Exception as e:
                        print(f"  [MKI] Warning: failed to compute key/value: {e}")
                        _mki_keys = None
                        _mki_values = None

                # predict noise model_output
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=device):
                    noise_pred = local_transformer(
                        x=latent_model_input,
                        context=in_prompt_embeds,
                        t=timestep,
                        seq_len=seq_len,
                        y=y,
                        y_camera=control_camera_latents_input,
                        map_tokens=getattr(self, "_map_tokens", None),
                        map_guidance=_smg_signal,
                        map_latent=_map_latent_signal,
                        map_kv_keys=_mki_keys,
                        map_kv_values=_mki_values,
                        semantic_layout_labels=(layout_labels_input if semantic_layout_features is None else None),
                        semantic_layout_features=(layout_labels_input if semantic_layout_features is not None else None),
                        semantic_layout_progress=layout_progress_input,
                        map_strength=strength,
                    )

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # === Layout guidance (Route B): sampling-time screen-occupancy steering ===
                # Steers the velocity so the predicted clean latent x0_hat = x_t - sigma*v
                # reads (under the layout critic E) as the external occupancy guide.
                # Output-space classifier guidance: no backprop through the DiT — v is the
                # only grad-carrying tensor. The gradient is RMS-normalized so lambda is
                # the fraction of velocity RMS spent on layout per step (dose-controlled).
                # Frame 0 is excluded: it is re-locked to the start-image latent below.
                _lg_cfg = getattr(self, "_layout_guidance_config", None)
                _lg_end = int(_lg_cfg.get("end_step", -1)) if _lg_cfg is not None else -1
                if _lg_cfg is not None and float(_lg_cfg.get("lambda", 0.0)) > 0 and i >= int(_lg_cfg.get("start_step", 0)) \
                        and (_lg_end < 0 or i < _lg_end) and _lg_cfg.get("guide") is not None:
                    try:
                        # __call__ runs under @torch.no_grad(); requires_grad_ alone
                        # does not build a graph there — autograd must be re-enabled.
                        with torch.enable_grad():
                            _critic = _lg_cfg["critic"]
                            _guide = _lg_cfg["guide"].to(device=latents.device, dtype=torch.float32)
                            _cw = _lg_cfg.get("channel_weights")
                            if _cw is not None:
                                _cw = _cw.to(device=latents.device, dtype=torch.float32)
                            _sigma = float(t) / self.scheduler.config.num_train_timesteps
                            _v = noise_pred.detach().float().requires_grad_(True)
                            _x0 = latents.detach().float() - _sigma * _v
                            _B, _C, _F, _H, _W = _x0.shape
                            _xf = _x0.permute(0, 2, 1, 3, 4).reshape(_B * _F, _C, _H, _W)
                            _pred = _critic(_xf).view(_B, _F, -1, _H, _W)[:, 1:]
                            _tgt = _guide.unsqueeze(0)[:, 1:_F]
                            _diff = (_pred - _tgt) ** 2
                            if _cw is not None:
                                _diff = _diff * _cw.view(1, 1, -1, 1, 1)
                            # guide_mask (world-canvas read): (F,1,H,W) in {0,1};
                            # 0 = canvas has no observation there -> exclude from
                            # the loss instead of pulling the frame toward 0.
                            _gm = _lg_cfg.get("guide_mask")
                            if _gm is not None:
                                _gm = _gm.to(device=latents.device, dtype=torch.float32)
                                _gm = _gm.unsqueeze(0)[:, 1:_F]
                                if _gm.shape[-2:] != _diff.shape[-2:]:
                                    _gm = torch.nn.functional.interpolate(
                                        _gm, size=_diff.shape[-2:], mode="nearest"
                                    )
                                _diff = _diff * _gm
                                _n_valid = _gm.sum().clamp_min(1.0) * _diff.shape[2]
                                _loss = _diff.sum() / _n_valid
                            else:
                                _loss = _diff.mean()
                            _g = torch.autograd.grad(_loss, _v)[0]
                        _v_rms = _v.detach().pow(2).mean().sqrt().clamp_min(1e-8)
                        _g_rms = _g.pow(2).mean().sqrt().clamp_min(1e-8)
                        if not _lg_cfg.get("_logged_once"):
                            _lg_cfg["_logged_once"] = True
                            print(f"  [LG] active: lambda={_lg_cfg['lambda']} step={i} loss={float(_loss):.5f} |g|rms={float(_g_rms):.2e} |v|rms={float(_v_rms):.2e}")
                        noise_pred = noise_pred - (float(_lg_cfg["lambda"]) * _v_rms / _g_rms * _g).to(noise_pred.dtype)
                    except Exception as _lg_e:
                        print(f"  [LG] layout guidance skipped at step {i}: {_lg_e}")

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # anchor10: per-step hard re-lock of covered pixels onto the
                # prior's FM trajectory. At each step the scheduler has moved
                # ALL pixels along the predicted velocity from sigma_{i+s0}
                # to sigma_{i+s0+1}; we OVERWRITE the covered pixels back to
                # where the deterministic prior path says they should be at
                # the POST-step sigma level. The prior path is
                # x = (1-sigma_{k})*prior + sigma_{k}*eps_fix (same blend as
                # the initialisation, evaluated at the post-step sigma). Frame
                # 0 stays locked to start_image via wmask[:,:,0]=0. This is
                # the frame-0 start-image mechanism generalised to all covered
                # pixels — the only injection mode ever measured to preserve
                # sharpness (1022.7 vs bank 1018.4).
                if (_anchor_active and _anchor_warm is not None
                        and float(self.scheduler.sigmas[i + _warm_s0 + 1].item()) >= _anchor_release):
                    _si = i + _warm_s0 + 1  # post-step sigma index
                    _sig = float(self.scheduler.sigmas[_si].item())
                    _anchored = (1.0 - _sig) * (_anchor_wmask * _anchor_warm) + _sig * _anchor_eps
                    latents = _anchor_wmask * _anchored + (1.0 - _anchor_wmask) * latents
                # release: once sigma drops below _anchor_release the trajectory
                # runs FREE — the final low-sigma steps repaint texture over the
                # anchored layout instead of dumping the raw prior latent.

                if self.vae.spatial_compression_ratio >= 16 and mask[:, :, 1, :, :].any():
                    latents = (1 - mask) * start_image_latentes + mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if comfyui_progressbar:
                    pbar.update(1)

        _dit_sec = time.time() - _denoise_t0
        print(f"[world_warm_timing] seg={_warm_seg} warm={int(_warm_s0 > 0)} "
              f"steps_run={len(timesteps)} dit_sec={_dit_sec:.1f}", flush=True)

        # Capture clean latent for world latent cache / R1c pregeneration.
        # Stored on CPU to avoid GPU memory accumulation across segments.
        self._last_clean_latent = latents.detach().cpu()

        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanPipelineOutput(videos=video)

# MKI pipeline patch applied
