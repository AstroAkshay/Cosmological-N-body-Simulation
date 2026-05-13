from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension(
    name="update_velocity",
    sources=["code/update_v.pyx", "code/update_v_core.c"],
    include_dirs=[np.get_include(), "."],
    extra_compile_args=["/O2", "/openmp"],
)

setup(
    name="update_velocity",
    ext_modules=cythonize([ext], language_level=3),
)
