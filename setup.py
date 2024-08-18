from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="my_cython_package.parallel_example",
        sources=["my_cython_package/parallel_example.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-fopenmp"],   # Enable OpenMP
        extra_link_args=["-fopenmp"],      # Link against OpenMP
    )
]

setup(
    name="my_cython_package",
    version="0.1",
    packages=["my_cython_package"],
    ext_modules=cythonize(extensions),
    install_requires=["cython", "numpy"],
)
