gpt-oss-test/custom_kernels_design.md
# Custom MXFP4 Kernel Implementation Design

This document outlines the design for using custom MXFP4 kernels by bypassing the default `transformers` loading logic. Instead of relying on the hardcoded `replace_with_mxfp4_linear` integration, we manually replace the expert layers after loading the model in full precision (bfloat16).

## Overview

The strategy involves three main phases:
1.  **Safe Loading**: Load the model with forced dequantization to obtain standard bfloat16 weights.
2.  **Custom Kernel Acquisition**: Manually load the Triton kernels from a specified source (e.g., a local file or a specific Hugging Face Hub repository).
3.  **In-Place Replacement**: Iterate through the model, re-quantize the weights using the custom kernels, and swap the standard expert layers with a custom implementation.

## Prerequisites

-   The model must be loadable in `bfloat16` (sufficient RAM required).
-   Access to the `transformers.integrations.hub_kernels` utility for loading kernels.
-   Access to `transformers.integrations.mxfp4` utilities (`quantize_to_mxfp4`, `swizzle_mxfp4`) if reusing standard quantization logic, or equivalent custom functions.

## Reference Implementations

Before implementing custom kernels, it is useful to inspect the existing reference implementations downloaded by `transformers` or wrapped by `vllm`.

-   **Transformers (Hugging Face Hub Cache):**
    The actual Triton source code downloaded by `transformers` is located in the Hugging Face hub cache.
    Path: `/home/compute/.cache/huggingface/hub/models--kernels-community--triton_kernels/snapshots/fe4ef5eba8c97556f74511321f49f173f430fd83/build/torch-universal/triton_kernels`

    Relevant files:
    -   `matmul_ogs.py`: Entry point for matrix multiplication.
    -   `matmul_ogs_details/_matmul_ogs.py`: Core kernel implementation.

-   **vLLM (Triton Wrapper):**
    vLLM provides a wrapper around the `triton_kernels` package (if installed).
    Path: `.venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py`

## Detailed Implementation Steps

### 1. Load Model with Forced Dequantization

To prevent the library from injecting the default (potentially buggy) kernels, we force the model to load in bfloat16.

```python
from transformers import AutoConfig, AutoModelForCausalLM
import torch

model_id = "openai/gpt-oss-20b"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# Force dequantization to load as bfloat16
if hasattr(config, "quantization_config"):
    # Depending on config structure, set dequantize=True
    if isinstance(config.quantization_config, dict):
        config.quantization_config["dequantize"] = True
    else:
        config.quantization_config.dequantize = True

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # Load on CPU first to save GPU memory for quantization
    trust_remote_code=True
)
```

### 2. Define Custom Expert Layer

We define a `CustomMxfp4Experts` class that mimics the interface of `GptOssExperts` but uses our custom kernels for the forward pass.

```python
import torch.nn as nn

class CustomMxfp4Experts(nn.Module):
    def __init__(self, config, kernels_hub):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size
        self.kernels_hub = kernels_hub
        
        # Initialize storage for quantized weights (shape depends on swizzling layout)
        # Note: Actual shapes are determined during the conversion step
        self.gate_up_proj = nn.Parameter(torch.empty(0), requires_grad=False)
        self.gate_up_proj_scale = nn.Parameter(torch.empty(0), requires_grad=False)
        
        self.down_proj = nn.Parameter(torch.empty(0), requires_grad=False)
        self.down_proj_scale = nn.Parameter(torch.empty(0), requires_grad=False)

        # Configuration for activation
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)

    def forward(self, hidden_states, routing_data, gather_idx, scatter_idx):
        # Unpack kernels from the hub object
        matmul_ogs = self.kernels_hub.matmul_ogs.matmul_ogs
        FusedActivation = self.kernels_hub.matmul_ogs.FusedActivation
        FnSpecs = self.kernels_hub.matmul_ogs.FnSpecs
        swiglu_fn = self.kernels_hub.swiglu.swiglu_fn

        # 1. Activation
        act = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), 
            (self.alpha, self.limit), 
            2
        )

        # 2. Gate/Up Projection
        # Note: Precision configs might need to be reconstructed or passed in
        intermediate = matmul_ogs(
            hidden_states,
            self.gate_up_proj,
            None, # bias is usually None for these layers or fused
            routing_data,
            gather_indx=gather_idx,
            # precision_config=..., # Pass scale tensor if kernel expects it separate or wrapped
            gammas=None,
            fused_activation=act,
        )

        # 3. Down Projection
        output = matmul_ogs(
            intermediate,
            self.down_proj,
            None,
            routing_data,
            scatter_indx=scatter_idx,
            # precision_config=..., 
            gammas=routing_data.gate_scal,
        )
        
        return output
```

### 3. Execution: Re-quantization and Replacement

We iterate over the model, convert the weights using `quantize_to_mxfp4` and `swizzle_mxfp4` (invoked via the custom kernel hub to ensure compatibility), and replace the modules.

```python
from transformers.integrations.hub_kernels import get_kernel
from transformers.integrations.mxfp4 import quantize_to_mxfp4, swizzle_mxfp4

# 1. Load your CUSTOM kernel repo
# Replace 'kernels-community/triton_kernels' with your fixed repo or local path
CUSTOM_REPO_ID = "my-custom-org/fixed-triton-kernels" 
custom_kernels_hub = get_kernel(CUSTOM_REPO_ID)

def convert_and_replace(model, custom_hub):
    device = torch.device("cuda") # Quantization usually requires GPU for Triton

    # Identify the class type to replace (from the loaded model)
    # Note: You might need to import GptOssExperts from the remote code or inspect module type
    target_class_name = "GptOssExperts"

    for name, module in model.named_modules():
        if module.__class__.__name__ == target_class_name:
            print(f"Replacing module: {name}")
            
            # 1. Create new layer
            new_layer = CustomMxfp4Experts(model.config, custom_hub)
            
            # 2. Convert Gate/Up Projection
            # Get original BF16 weights
            w_gate_up = module.gate_up_proj.weight.to(device)
            
            # Quantize
            q_gate_up, s_gate_up = quantize_to_mxfp4(w_gate_up, custom_hub)
            
            # Swizzle (optimize layout for kernel)
            # Transpose is often required before swizzling depending on kernel expectations
            q_gate_up_swizzled, s_gate_up_swizzled = swizzle_mxfp4(
                q_gate_up.transpose(-1, -2), # Ensure correct shape for swizzle
                s_gate_up, 
                custom_hub
            )
            
            # Assign to new layer
            new_layer.gate_up_proj = nn.Parameter(q_gate_up_swizzled, requires_grad=False)
            new_layer.gate_up_proj_scale = nn.Parameter(s_gate_up_swizzled, requires_grad=False)

            # 3. Convert Down Projection
            w_down = module.down_proj.weight.to(device)
            q_down, s_down = quantize_to_mxfp4(w_down, custom_hub)
            q_down_swizzled, s_down_swizzled = swizzle_mxfp4(
                q_down.transpose(-1, -2), 
                s_down, 
                custom_hub
            )
            
            new_layer.down_proj = nn.Parameter(q_down_swizzled, requires_grad=False)
            new_layer.down_proj_scale = nn.Parameter(s_down_swizzled, requires_grad=False)
            
            # 4. Replace in parent module
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_layer)
            
            # Clean up VRAM
            del w_gate_up, w_down
            torch.cuda.empty_cache()

# Execute replacement
convert_and_replace(model, custom_kernels_hub)

# Move model to GPU for inference
model.to("cuda")
```

## Summary
By manually controlling the quantization and replacement loop, we:
1.  Avoid the `transformers` hardcoded `get_kernel` call.
2.  Ensure that the weights are quantized specifically for the layout expected by our *custom* kernels (passed via `custom_hub`).
3.  Inject our fixed/custom Triton implementation into the model architecture.