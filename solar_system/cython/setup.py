#Compile with "python3 setup.py build_ext --inplace"

from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "py_solar_system",
        sources=["py_solar_system.pyx"],
        extra_compile_args=["-Ofast", "-mtune=native", "-std=c++11"],
        language="c++",
    )
]

setup(name="PySolarSystem",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"})
)