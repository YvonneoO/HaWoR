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
    cmdclass={ 'build_ext' : BuildExtension }
)

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'), 
                *EIGEN_INCLUDE_DIRS],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp', 
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-O2'], 
                # Let PyTorch honor TORCH_CUDA_ARCH_LIST instead of pinning old SMs.
                'nvcc': ['-O2']
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
