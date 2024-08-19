from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension("geovox.utility", ["lib/utility.pyx"]),
    Extension("geovox.particles", ["lib/particles.pyx", "lib/utility.pyx"])
]

setup(
    name="geovox",
    version="0.1",
    packages=["geovox"],
    ext_modules=cythonize(extensions),
    install_requires=["cython"],
)
