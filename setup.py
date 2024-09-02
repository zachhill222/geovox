from setuptools import setup, Extension
from Cython.Build import cythonize


# define extensions
extensions = [
    Extension(
        name="geovox.linalg.vector",
        sources=["geovox/linalg/vector.pyx", "c_geovox/src/point.cpp"],
        language="c++",
        include_dirs=["c_geovox/include"]
    ),
    Extension(
        name="geovox.linalg.matrix",
        sources=["geovox/linalg/matrix.pyx"],
        language="c++",
        include_dirs=["c_geovox/include"]
    ),
    Extension(
        name="geovox.linalg.solvers",
        sources=["geovox/linalg/solvers.pyx"],
        language="c++",
        include_dirs=["c_geovox/include"]
    ),
    Extension(
        name="geovox.util.box",
        sources=["geovox/util/box.pyx"],
        language="c++",
        include_dirs=["c_geovox/include"]
    ),
    Extension(
        name="geovox.util.quaternion",
        sources=["geovox/util/quaternion.pyx"],
        language="c++",
        include_dirs=["c_geovox/include"]
    ),
    Extension(
        name="geovox.util.optimize",
        sources=["geovox/util/optimize.pyx"],
        language="c++",
        include_dirs=["c_geovox/include"]
    ),
    Extension(
        name="geovox.geometry.particles",
        sources=["geovox/geometry/particles.pyx"],
        language="c++",
        include_dirs=["c_geovox/include"]
    ),
    Extension(
        name="geovox.geometry.assembly",
        sources=["geovox/geometry/assembly.pyx"],
        language="c++",
        include_dirs=["c_geovox/include"]
    ),
    Extension(
        name="geovox.vtkutil",
        sources=["geovox/vtkutil.pyx"],
        language="c++",
        include_dirs=["c_geovox/include"]
    )
]

setup(
    name="geovox",
    version="0.1",
    packages=["geovox", "geovox.util", "geovox.geometry", "geovox.linalg"],
    ext_modules=cythonize(extensions)
)
