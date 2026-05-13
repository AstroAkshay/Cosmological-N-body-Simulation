from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

extensions = [
    Extension(
        "calpow_module",
        sources=[
            os.path.join("code", "calpow_wrapper.pyx"),
            os.path.join("code", "calpow_from_k.c"),
        ],
        include_dirs=[
            np.get_include(),
            r"C:\Users\aksha\vcpkg\installed\x64-windows\include",
        ],
        library_dirs=[
            r"C:\Users\aksha\vcpkg\installed\x64-windows\lib",
        ],
        libraries=["fftw3f"],
        language="c",
    )
]

setup(
    name="calpow_module",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3}),
)
