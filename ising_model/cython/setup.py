#Compile with "python3 setup.py build_ext --inplace"

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "ising2d",
        sources=["ising.pyx"],
        extra_compile_args=["-O3"],
        include_dirs=[numpy.get_include()]
    )
]

setup(name="Ising2D",
    ext_modules=cythonize(ext_modules)
)