# Install lietorch first (one package per setuptools run; pip -e cannot handle two setup() calls):
#   cd thirdparty/lietorch && pip install -e . --no-build-isolation
# Then from this directory:
#   pip install -e . --no-build-isolation
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))
EIGEN_INCLUDE_DIRS = [
    osp.join(ROOT, 'thirdparty/eigen'),
    '/usr/include/eigen3',
]

setup(
    name='droid_backends',
    ext_modules=[
        CUDAExtension('droid_backends',
            include_dirs=EIGEN_INCLUDE_DIRS,
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                # Let PyTorch honor TORCH_CUDA_ARCH_LIST instead of pinning old SMs.
                'nvcc': ['-O3']
            }),
    ],
    cmdclass={'build_ext': BuildExtension}
)
