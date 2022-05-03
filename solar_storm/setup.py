from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "py_solar_storm",
        sources=["py_solar_storm.pyx"],
        extra_compile_args=["-Xpreprocessor" ,"-fopenmp", "-Ofast", "-std=c++11"],
        extra_link_args=["-lomp"],
        language="c++",
    )
]

setup(name="PySolarStorm",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level' : "3"},),
)