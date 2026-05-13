from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

vcpkg_root = r"C:\Users\aksha\vcpkg\installed\x64-windows"

gradphi_ext = Extension(
    name="gradphi",
    sources=["code/gradphi.pyx", "code/grad_phi.c"],
    include_dirs=[np.get_include(), ".", os.path.join(vcpkg_root, "include")],
    library_dirs=[os.path.join(vcpkg_root, "lib")],
    libraries=["fftw3f"],  # make sure this matches your FFTW lib name
    extra_compile_args=["/O2", "/openmp"],
    extra_link_args=[],
)

setup(
    name="gradphi",
    ext_modules=cythonize([gradphi_ext], language_level="3"),
)
