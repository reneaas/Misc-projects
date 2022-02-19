#Compile with "python3 setup.py build_ext --inplace"

from setuptools import setup, Extension

from Cython.Build import cythonize

ext_modules = [
    Extension(
        "solar_system",
        sources=["py_solar_system.pyx"],
        extra_compile_args=["-O2"],
    )
]

setup(name="SolarSystem",
    ext_modules=cythonize(ext_modules)
)