from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Define paths relative to this setup.py file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CUTLASS_INCLUDE = os.path.abspath(os.path.join(ROOT_DIR, "..", "deps", "cutlass", "include"))
LOCAL_INCLUDE = os.path.join(ROOT_DIR, "include")

print(f"Building with CUTLASS include: {CUTLASS_INCLUDE}")

setup(
    name="matmul_mxfp4_cutlass_ext",
    ext_modules=[
        CUDAExtension(
            name="matmul_mxfp4_cutlass_ext",
            sources=["matmul_mxfp4_cutlass.cu"],
            include_dirs=[
                LOCAL_INCLUDE,
                CUTLASS_INCLUDE,
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
