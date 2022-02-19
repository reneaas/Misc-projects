from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "integrator",
        sources=["integrator.pyx"],
        extra_compile_args=["-Xpreprocessor" ,"-fopenmp", "-O3"],
        extra_link_args=["-lomp"]
    )
]

setup(name="Integrator",
    ext_modules=cythonize(ext_modules)
)