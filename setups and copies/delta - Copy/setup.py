from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "fftgrid_wrapper",
        sources=["code/fftgrid_wrapper.pyx", "code/fftgrid.c"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="fftgrid",
    ext_modules=cythonize(ext_modules),
)
