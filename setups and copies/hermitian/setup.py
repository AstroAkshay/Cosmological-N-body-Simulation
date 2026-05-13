from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="hermitian_wrapper",
    sources=["code/hermitian_wrapper.pyx", "code/hermitian_grid.c"],
    include_dirs=[numpy.get_include(), r"C:\Users\aksha\vcpkg\installed\x64-windows\include"],
    library_dirs=[r"C:\Users\aksha\vcpkg\installed\x64-windows\lib"],
    libraries=["fftw3"],  # adjust based on your .lib filename
    extra_compile_args=["/O2", "/openmp"],
    extra_link_args=[],
    language="c",
)

setup(
    name="hermitian_wrapper",
    ext_modules=cythonize([ext], language_level=3),
)
