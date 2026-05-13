from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="hermitian_wrapper",
        sources=["code/hermitian_wrapper.pyx", "code/hermitian_grid.c"],
        include_dirs=[np.get_include(), "."],
        language="c",
    )
]

setup(
    name="hermitian_wrapper",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
