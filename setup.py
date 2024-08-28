from setuptools import setup, Extension
from Cython.Build import cythonize


# define extensions
extensions = [
    Extension(
        name="geovox.utilities",
        sources=["geovox/utilities.pyx"]
    ),
    Extension(
        name="geovox.particles",
        sources=["geovox/particles.pyx"]
    ),
    Extension(
        name="geovox.vtkutil",
        sources=["geovox/vtkutil.pyx"]
    ),
    Extension(
        name="geovox.assembly",
        sources=["geovox/assembly.pyx"]
    ),
    Extension(
        name="geovox.optimize",
        sources=["geovox/optimize.pyx"])
]

setup(
    name="geovox",
    version="0.1",
    packages=["geovox"],
    ext_modules=cythonize(extensions),
    install_requires=["cython", "scipy"],
)
