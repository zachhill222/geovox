from setuptools import setup, Extension
from Cython.Build import cythonize


# define extensions
extensions = [
    Extension(
        name="geovox.utilities.vector3",
        sources=["geovox/utilities/vector3.pyx"]
    ),
    Extension(
        name="geovox.utilities.box",
        sources=["geovox/utilities/box.pyx"]
    ),
    Extension(
        name="geovox.utilities.quaternion",
        sources=["geovox/utilities/quaternion.pyx"]
    ),
    Extension(
        name="geovox.utilities.optimize",
        sources=["geovox/utilities/optimize.pyx"]
    ),
    Extension(
        name="geovox.geometry.particles",
        sources=["geovox/geometry/particles.pyx"]
    ),
    Extension(
        name="geovox.geometry.assembly",
        sources=["geovox/geometry/assembly.pyx"]
    ),
    Extension(
        name="geovox.vtkutil",
        sources=["geovox/vtkutil.pyx"]
    )
]

setup(
    name="geovox",
    version="0.1",
    packages=["geovox", "geovox.utilities", "geovox.geometry"],
    ext_modules=cythonize(extensions)
)
