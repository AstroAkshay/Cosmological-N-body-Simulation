from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    "cic_wrap",
    sources=["code/cic_wrap.pyx", "code/cic.c"],
    include_dirs=["code", np.get_include()],
    extra_compile_args=["/openmp", "/O2"],    # Compiler only
    # extra_link_args=["/openmp"],           # REMOVE THIS LINE
)

setup(
    ext_modules=cythonize([ext], compiler_directives={"language_level": "3"})
)
