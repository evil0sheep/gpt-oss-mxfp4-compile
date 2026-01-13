import os
import sys

# Add deps/transformers/src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
transformers_path = os.path.join(current_dir, "deps", "transformers", "src")
sys.path.append(transformers_path)

import torch
from transformers.integrations.hub_kernels import get_kernel
from transformers.utils import (
    is_accelerate_available,
    is_kernels_available,
    is_torch_available,
    is_triton_available,
)


def main():
    print(f"Torch available: {is_torch_available()}")
    if is_torch_available():
        print(f"Torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA capabilities: {torch.cuda.get_device_capability()}")

    print(f"Accelerate available: {is_accelerate_available()}")

    print(f"Triton available: {is_triton_available()}")
    if is_triton_available():
        import triton

        print(f"Triton version: {triton.__version__}")

    print(f"Kernels available: {is_kernels_available()}")
    if is_kernels_available():
        import kernels

        print(f"Kernels version: {kernels.__version__}")

    try:
        print("Attempting to load triton kernels from hub...")
        triton_kernels_hub = get_kernel("kernels-community/triton_kernels")
        print("Successfully loaded triton kernels hub")
        # Try to inspect available kernels/utils
        print(f"Hub contents: {dir(triton_kernels_hub)}")

    except Exception as e:
        print(f"Failed to load triton kernels: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
