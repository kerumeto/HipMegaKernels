import os
import subprocess
from pybind11.setup_helpers import build_ext
import pybind11
from setuptools import setup, Extension

# Environment variables
THUNDERKITTENS_ROOT = os.environ.get('THUNDERKITTENS_ROOT', '')
MEGAKERNELS_ROOT = os.environ.get('MEGAKERNELS_ROOT', '')
PYTHON_VERSION = os.environ.get('PYTHON_VERSION', '3.10') 
ROCM_HOME = os.environ.get('ROCM_HOME', '/opt/rocm')

# Target GPU (default to MI300X/gfx942)
TARGET = os.environ.get('TARGET_GPU', 'MI300X') 

# Source file
SRC = 'src/{{PROJECT_NAME_LOWER}}.cu'

# Get Python include directory
def get_python_include():
    try:
        import sysconfig
        return sysconfig.get_path('include')
    except ImportError:
        return ''

# Base HIPCC flags
# FIX: Removed -Xcompiler wrapper. Passed flags directly.
hipcc_flags = [
    '-DNDEBUG',
    '-O3',
    '-std=c++20',
    '-fPIC',
    '-w',
    '-D__HIP_PLATFORM_AMD__',
    '--use_fast_math',
    '-Wno-psabi',            # FIXED: Passed directly
    '-fno-strict-aliasing',  # FIXED: Passed directly
    '-lrt',
    '-lpthread',
    '-ldl',
    '-lhipblas',
    '-lrocblas',
    '-shared',
    f'-lpython{PYTHON_VERSION}'
]

# Include directories
include_dirs = [
    f'{THUNDERKITTENS_ROOT}/include',
    f'{MEGAKERNELS_ROOT}/include',
    f'{ROCM_HOME}/include',
    pybind11.get_include(),
    get_python_include()
]

# Get python config flags
def get_python_config_flags():
    try:
        ldflags = subprocess.check_output(['python3-config', '--ldflags']).decode().strip().split()
        return ldflags
    except subprocess.CalledProcessError:
        return []

# Add python config flags
hipcc_flags.extend(get_python_config_flags())

# Conditional setup based on target GPU
if TARGET in ['MI300', 'MI300X', 'gfx942']:
    hipcc_flags.extend(['--offload-arch=gfx942', '-DKITTENS_MI300X'])
elif TARGET in ['MI200', 'MI250', 'MI250X', 'gfx90a']:
    # Note: Use MI250 or MI200 to trigger proper macros from Makefile logic
    hipcc_flags.extend(['--offload-arch=gfx90a', '-DKITTENS_MI250']) 
elif TARGET == 'native':
    hipcc_flags.extend(['--offload-arch=native'])
else:
    # Default fallback
    hipcc_flags.extend(['--offload-arch=gfx942', '-DKITTENS_MI300X'])

# Custom build extension class to use hipcc
class HipExtension(Extension):
    def __init__(self, name, sources, **kwargs):
        super().__init__(name, sources, **kwargs)

class HipBuildExt(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, HipExtension):
            self.build_hip_extension(ext)
        else:
            super().build_extension(ext)
    
    def build_hip_extension(self, ext):
        # Locate hipcc
        hipcc = os.environ.get('HIPCC', os.path.join(ROCM_HOME, 'bin/hipcc'))
        
        # Get the output file path
        ext_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        
        # Build the command
        cmd = [hipcc] + ext.sources + hipcc_flags + ['-o', ext_path]
        
        # Add include directories
        for include_dir in include_dirs:
            cmd.extend(['-I', include_dir])
        
        print(f"Building HIP extension with command: {' '.join(cmd)}")
        subprocess.check_call(cmd)

# Define the extension
ext_modules = [
    HipExtension(
        '{{PROJECT_NAME_LOWER}}',
        sources=[SRC],
    )
]

setup(
    name='{{PROJECT_NAME_LOWER}}',
    ext_modules=ext_modules,
    cmdclass={'build_ext': HipBuildExt},
    zip_safe=False,
    python_requires=">=3.6",
)

# import os
# import subprocess
# from pybind11.setup_helpers import build_ext
# import pybind11
# from setuptools import setup, Extension

# # Environment variables
# THUNDERKITTENS_ROOT = os.environ.get('THUNDERKITTENS_ROOT', '')
# MEGAKERNELS_ROOT = os.environ.get('MEGAKERNELS_ROOT', '')
# PYTHON_VERSION = os.environ.get('PYTHON_VERSION', '3.13')

# # Target GPU (default to HOPPER)
# TARGET = os.environ.get('TARGET_GPU', 'HOPPER') # or BLACKWELL

# # Source file
# SRC = 'src/{{PROJECT_NAME_LOWER}}.cu'

# # Get Python include directory
# def get_python_include():
#     try:
#         python_include = subprocess.check_output(['python', '-c', "import sysconfig; print(sysconfig.get_path('include'))"]).decode().strip()
#         return python_include
#     except subprocess.CalledProcessError:
#         return ''

# # Base NVCC flags
# nvcc_flags = [
#     '-DNDEBUG',
#     '-Xcompiler=-fPIE',
#     '--expt-extended-lambda',
#     '--expt-relaxed-constexpr',
#     '-Xcompiler=-Wno-psabi',
#     '-Xcompiler=-fno-strict-aliasing',
#     '--use_fast_math',
#     '-forward-unknown-to-host-compiler',
#     '-O3',
#     '-Xnvlink=--verbose',
#     '-Xptxas=--verbose',
#     '-Xptxas=--warn-on-spills',
#     '-std=c++20',
#     '-x', 'cu',
#     '-lrt',
#     '-lpthread',
#     '-ldl',
#     '-lcuda',
#     '-lcudadevrt',
#     '-lcudart_static',
#     '-lcublas',
#     '-lineinfo',
#     '-shared',
#     '-fPIC',
#     f'-lpython{PYTHON_VERSION}'
# ]

# # Include directories
# include_dirs = [
#     f'{THUNDERKITTENS_ROOT}/include',
#     f'{MEGAKERNELS_ROOT}/include',
#     pybind11.get_include(),
#     get_python_include()
# ]

# # Get python config flags
# def get_python_config_flags():
#     try:
#         ldflags = subprocess.check_output(['python3-config', '--ldflags']).decode().strip().split()
#         return ldflags
#     except subprocess.CalledProcessError:
#         return []

# # Add python config flags
# nvcc_flags.extend(get_python_config_flags())

# # Conditional setup based on target GPU
# if TARGET == 'HOPPER':
#     nvcc_flags.extend(['-DKITTENS_HOPPER', '-arch=sm_90a'])
# elif TARGET == 'BLACKWELL':
#     nvcc_flags.extend(['-DKITTENS_HOPPER', '-DKITTENS_BLACKWELL', '-arch=sm_100a'])
# else:
#     raise ValueError(f"Invalid target: {TARGET}")

# # Get python extension suffix
# def get_extension_suffix():
#     try:
#         suffix = subprocess.check_output(['python3-config', '--extension-suffix']).decode().strip()
#         return suffix
#     except subprocess.CalledProcessError:
#         return '.so'

# # Custom build extension class to use nvcc
# class CudaExtension(Extension):
#     def __init__(self, name, sources, **kwargs):
#         super().__init__(name, sources, **kwargs)

# class CudaBuildExt(build_ext):
#     def build_extension(self, ext):
#         if isinstance(ext, CudaExtension):
#             self.build_cuda_extension(ext)
#         else:
#             super().build_extension(ext)
    
#     def build_cuda_extension(self, ext):
#         nvcc = os.environ.get('NVCC', 'nvcc')
        
#         # Get the output file path
#         ext_path = self.get_ext_fullpath(ext.name)
        
#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        
#         # Build the nvcc command
#         cmd = [nvcc] + ext.sources + nvcc_flags + ['-o', ext_path]
        
#         # Add include directories
#         for include_dir in include_dirs:
#             cmd.extend(['-I', include_dir])
        
#         print(f"Building CUDA extension with command: {' '.join(cmd)}")
        
#         # Execute the command
#         subprocess.check_call(cmd)

# # Define the extension
# ext_modules = [
#     CudaExtension(
#         '{{PROJECT_NAME_LOWER}}',
#         sources=[SRC],
#     )
# ]

# setup(
#     name='{{PROJECT_NAME_LOWER}}',
#     ext_modules=ext_modules,
#     cmdclass={'build_ext': CudaBuildExt},
#     zip_safe=False,
#     python_requires=">=3.6",
# )
