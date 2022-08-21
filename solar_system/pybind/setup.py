#Compile with "python3 setup.py build_ext --inplace"

from setuptools import setup, Extension
from Cython.Build import cythonize

from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
    'solar_system',
    ["solar_system.cpp", "wrapper.cpp"],
    language='c++',
    extra_compile_args = ['-std=c++14'],
    ),
]


setup(name="SolarSystem",
    ext_modules=ext_modules,
)

