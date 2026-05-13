from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="get_phi_k",
    sources=[
        "code/get_phi_k.pyx",
        "code/poisson_kernel.c",
    ],
    include_dirs=[np.get_include(), "code"],
    extra_compile_args=["/O2", "/openmp"],  # ← ADD THESE
)

setup(
    ext_modules=cythonize([ext], language_level=3),  # ← [ext] list!
)
