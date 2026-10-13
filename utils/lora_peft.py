# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
import torch
import peft
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig
)


def configure_lora_for_model_bk(transformer, model_name, lora_config, is_main_process=True):
    """Configure LoRA for a WanDiffusionWrapper model
    
    Args:
        transformer: The transformer model to apply LoRA to
        model_name: 'generator' or 'fake_score'
        lora_config: LoRA configuration
        is_main_process: Whether this is the main process (for logging)
    
    Returns:
        lora_model: The LoRA-wrapped model
    """
    # Find all Linear modules in WanAttentionBlock modules
    target_linear_modules = set()
    
    # Define the specific modules we want to apply LoRA to
    if model_name == 'generator':
        adapter_target_modules = ['CausalWanAttentionBlock']
    elif model_name == 'fake_score':
        adapter_target_modules = ['WanAttentionBlock']
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)
    
    target_linear_modules = list(target_linear_modules)
    
    if is_main_process:
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        if getattr(lora_config, 'verbose', False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")
    
    # Create LoRA config
    adapter_type = lora_config.get('type', 'lora')
    if adapter_type == 'lora':
        peft_config = peft.LoraConfig(
            r=lora_config.get('rank', 16),
            lora_alpha=lora_config.get('alpha', None) or lora_config.get('rank', 16),
            lora_dropout=lora_config.get('dropout', 0.0),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
    
    # Apply LoRA to the transformer
    lora_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()
    
    return lora_model

def configure_lora_for_model(transformer, model_name, lora_config, is_main_process=True):
    """Configure LoRA for a WanDiffusionWrapper model.

    When ``lora_config["block_start"]`` is set (e.g. 20), LoRA is applied
    only to DiT blocks whose index >= block_start.  Early-block LoRA layers
    produce 1e13-1e14 gradients (each transformer block amplifies ~3-5x),
    and clip_grad_norm_ on the combined norm then shrinks the well-behaved
    adapter/late-block gradients to near-zero.  Restricting LoRA to late
    blocks (matching the RRSC injection layers 20,29) keeps all trainable
    gradients in the same healthy range so the optimizer can learn.
    """
    # Find all Linear modules in WanAttentionBlock modules
    target_linear_modules = set()

    adapter_exclude_modules = [] #['PropeSelfAttention']
    block_start = int(lora_config.get('block_start', 0))

    # Define the specific modules we want to apply LoRA to
    if model_name == 'generator':
        if 'Causal' in transformer.__class__.__name__ :
            adapter_target_modules = ['CausalWanAttentionBlock']
        else:
            adapter_target_modules = ['WanAttentionBlock']
    elif model_name == 'fake_score':
        adapter_target_modules = ['WanAttentionBlock']
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            # Filter by block index when block_start > 0.
            # Block names look like "blocks.0", "blocks.1", ... "blocks.29".
            if block_start > 0:
                tail = name.rsplit('.', 1)[-1]
                if not tail.isdigit() or int(tail) < block_start:
                    continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                # 跳过被排除模块（如 PropeSelfAttention）内部的 Linear 层
                if submodule.__class__.__name__ in adapter_exclude_modules:
                    continue
                if isinstance(submodule, torch.nn.Linear):
                    # 检查该 Linear 是否位于某个被排除模块的子树下
                    is_excluded = False
                    for excluded_name, excluded_module in module.named_modules(prefix=name):
                        if excluded_module.__class__.__name__ in adapter_exclude_modules:
                            if full_submodule_name.startswith(excluded_name + "."):
                                is_excluded = True
                                break
                    if not is_excluded:
                        target_linear_modules.add(full_submodule_name)

    target_linear_modules = list(target_linear_modules)

    if is_main_process:
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers"
              + (f" (block_start={block_start})" if block_start > 0 else ""))
        if getattr(lora_config, 'verbose', False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")
    
    # Create LoRA config
    adapter_type = lora_config.get('type', 'lora')
    if adapter_type == 'lora':
        peft_config = peft.LoraConfig(
            r=lora_config.get('rank', 16),
            lora_alpha=lora_config.get('alpha', None) or lora_config.get('rank', 16),
            lora_dropout=lora_config.get('dropout', 0.0),
            target_modules=target_linear_modules,
        )
    else:
        raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
    
    # Apply LoRA to the transformer
    lora_model = peft.get_peft_model(transformer, peft_config)

    if is_main_process:
        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()
    
    return lora_model

def gather_lora_state_dict(lora_model):
    with FSDP.state_dict_type(
        lora_model,                  
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
    ):
        full = lora_model.state_dict()
    return get_peft_model_state_dict(lora_model, state_dict=full)


def load_lora_checkpoint(lora_model, lora_state_dict, model_name, is_main_process=True):
    """Load LoRA weights from state dict
    
    Args:
        lora_model: The LoRA-wrapped model
        lora_state_dict: LoRA state dict to load
        model_name: 'generator' or 'critic'
        is_main_process: Whether this is the main process (for logging)
    """
    if is_main_process:
        print(f"Loading LoRA {model_name} weights: {len(lora_state_dict)} keys in checkpoint")
    
    peft.set_peft_model_state_dict(lora_model, lora_state_dict)
    
    if is_main_process:
        print(f"LoRA {model_name} weights loaded successfully") 