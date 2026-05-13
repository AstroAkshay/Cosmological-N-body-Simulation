from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="cic_wrapper",
    sources=["code/cic_wrapper.pyx", "code/cic.c"],  # Include the C source here
    include_dirs=[np.get_include(), "."],  # "." if cic.h is in the same folder
    extra_compile_args=['/openmp'],          # MSVC OpenMP flag
    extra_link_args=['/openmp'],
    language="c",
)

setup(
    name="cic_wrapper",
    ext_modules=cythonize(ext, compiler_directives={'language_level': 3}),
)
