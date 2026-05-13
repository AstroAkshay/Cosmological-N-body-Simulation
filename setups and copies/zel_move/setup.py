from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "zel_move_wrap",
        sources=["code/zel_move_wrap.pyx", "code/zel_move.c"],
        include_dirs=["code", np.get_include()],
        extra_compile_args=["/openmp"],
        extra_link_args=["/openmp"],
    )
]

setup(
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)
