import os
import numpy as np
from setuptools import setup, Extension

vcpkg_root = r"C:\Users\aksha\vcpkg\installed\x64-windows"

fftw_module = Extension(
    "fftw.fftw3d",
    sources=["fftw/fftw3d.c"],
    include_dirs=[np.get_include(), os.path.join(vcpkg_root, "include")],
    library_dirs=[os.path.join(vcpkg_root, "lib")],  # <- correct path
    libraries=["fftw3"],
    extra_compile_args=["/O2"],
)

setup(
    name="fftw",
    version="1.0",
    packages=["fftw"],
    ext_modules=[fftw_module],
)
