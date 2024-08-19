from setuptools import setup, Extension
from Cython.Build import cythonize
from autogen_pxd import process_directory, delete_existing_pxd_files


# define extensions
extensions = [
    Extension(
        name="geovox.utility",
        sources=["geovox/utility.pyx"]
    ),
    Extension(
        name="geovox.particles",
        sources=["geovox/particles.pyx"]
    )
]

setup(
    name="geovox",
    version="0.1",
    packages=["geovox"],
    ext_modules=cythonize(extensions),
    install_requires=["cython"],
)
