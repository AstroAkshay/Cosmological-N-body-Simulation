from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="update_position",                    
    sources=["code/update_x_wrap.pyx", "code/update_x_core.c"],
    include_dirs=[np.get_include(), "code"],   
    extra_compile_args=["/O2", "/openmp"],     # ← ADD /openmp ONLY
)

setup(
    name="update_position",
    ext_modules=cythonize([ext], language_level=3),
)
