from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

ext = Extension(
    name="fftw.fftw_helper",
    sources=[
        os.path.join("fftw", "fftw_helper.pyx"),
        os.path.join("fftw", "fftw_helper_manual.c"),   # 🔥 IMPORTANT: add this line!
    ],
    include_dirs=[
        np.get_include(),
        r"C:\Users\aksha\vcpkg\installed\x64-windows\include",
    ],
    library_dirs=[
        r"C:\Users\aksha\vcpkg\installed\x64-windows\lib",
    ],
    libraries=["fftw3"],
    language="c",
)

setup(
    name="fftw",
    ext_modules=cythonize([ext], language_level=3),
)
