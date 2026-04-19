**This library is under active development and will frequently change**

# cgeovox
* A C++ library for voxel-based geometry and finite elements. 
* There are many ```assert()``` calls, so compiling with ```#define NDBUG``` after testing may increase performance.
* This library is primarily for 3D problems but should work for 2D as well, but there may be more bugs.



## Modules/Namespaces
### `gv::util`
Provides much of the base functionality of this library. `point.hpp`, `box.hpp`, `octree.hpp`, and `point_octree.hpp` are especially extensively used. `point_octree.hpp` provides a simple class and an example of how to extend the `BasicOctree` class. `plane.hpp` and `polytope.hpp` are used to help with the colision detection algorithm (`gv::geometry::collides_GJK` in `collisions.hpp`). `quaternion.hpp` is primarily used to handle the rotations of particles in `gv::geometry`. The commonly used classes are covered below.
- `point.hpp` Provides a templated class Point<dim,Scalar_t> which is extensively used in this library. This class is essentially a (mathematical) vector with operator overloading and positive/negative cone comparison. It is designed to be very easy to work with and has flexible constructors and copy/move assignments. The majority of this library uses Point<3,double> for points in 3D space and sometimes Point<3,size_t> for indices which will be converted to a point in space.
```C++
Point<3,double>{1,2,3} <= Point<3,double>{1,3,4}; //true
Point<3,double>{1,3,4} >= Point<3,double>{1,2,3}; //true
Point<3,double>{1,2,5} <= Point<3,double>{2,3,4}; //false
Point<3,double>{2,3,4} <= Point<3,double>{1,2,5}; //also false
Point<3,double>{1,2,3} *  Point<3,double>{1,2,3}; //Point<3,double>{1,4,9}
Point<3,double>{1,2,3} /  Point<3,double>{1,2,3}; //Point<3,double>{1,1,1}
std::cout << Point<3,double>{1,2,3};              //1 2 3 
```
- `box.hpp` Provides a templated class Box<dim,Scalar_t> which is extensively used in this library. This class is essentially two Point<dim,Scalar_t> objects _low and _high such that _low<_high is true. This object allows for testing if a point is contained in a box, if two boxes intersect, and getting vertices of the box (usually in vtk voxel/pixel) order. The majority of this library uses Box<3,double>. The primary constructor Box(Point,Point) will sort the provided points to get the compatible _low and _high vertices if possible. This class is primarily used as a bounding box for objects.
```C++
Box<3,double> bbox(Point<3,double>{1,0,1},Point<3,double>{0,1,0}); //same as: Box<3,double> bbox(Point<3,double>{0,0,0},Point<3,double>{1,1,1})
bbox[0]; //Point<3,double>{0,0,0}
bbox[7]; //Point<3,double>{1,1,1}
bbox.contains(Point<3,double>{0,0,0.5}); //true
bbox.contains(Point<3,double>{0,0,1.5}); //false
bbox.intersects(Box<3,double>{Point<3,double>{0.5,0.5,0.5}, Point<3,double>{1.5,1.5,1.5}); //true
std::cout << bbox; //(0 0 0 ) to (1 1 1 )
```
- `octree.hpp` Provides a templated container `BasicOctree<Data_t,dim,n_data>` (octree when `dim=3` or quadtree when `dim=2`) to allow finding objects located in space quickly. Each node of the octree has a bounding box (`Box<dim,double>`) assocaiated with it. The container has an interface similar to `std::vector` and was designed to hold objects which enclose a volume as well as objects that are located at a single point. This container has an array `Data_t* _data` which will be automatically resized and maintains the order that the data was added to the container (using `octree.push_back(Data_t)` or `octree.push_back(Data_t,size_t)`). The leaf nodes of the octree store (up to) `n_data` indices of type `size_t` corresponding to the objects in the `_data` array which 'intersect' with the bounding box of the leaf node and multiple leaf nodes may contain the same indices. Look at `point_octree.hpp` for an example of how to implement an extension of `BasicOctree` to store a specific `Data_t`. The extension must override the method `is_data_valid(Box<dim,double>, Data_t)`. If `Data_t` is an object that encloses a volume (say a sphere), then this method should test if the `Box` and `Data_t` intersect. For convex objects (i.e. particles found in `gv::geometry`), the [GJK algorithm](https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm) (contained in `collisions.hpp`) is quite useful for this. The elements of `_data` are unique in the sense that the `Data_` must have an `operator==` associated with them and `_data[i]==_data[j]` is true if and only if `i==j` is true.
>When `Data_t` encloses a volume, there is a danger when multiple objects overlap. If more than `n_data` objects overlap, it is possible that the octree will attempt divide to balance the data indices in the leaf nodes to be less than `n_data`. This may be impossible and result in a tree with infinite depth and crash the program. To avoid this, a constant `OCTREE_MAX_DEPTH` is defined in `compile_constants.hpp` with a default value of `32`.
```C++
size_t idx;
Data_t val;
int flag = octree.push_back(val, idx);
//if val was already contained in the octree, flag=0 and idx is updated so that octree[idx]==val is true
//if val was not contained in the octree and was successfully added, then flag=1 and idx is updated so that octree[idx]==val is true
//if val could not be added to the octree, then flag=-1.

//if the index of val is not relevant, the data can be inserted using:
flag = octree.push_back(val);
```

### `gv::mesh`
Provides some standard meshing utility. For now, only meshes of a single element type are allowed. Meshes can be written to a file in [VTK Legacy ASCII format](https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html) and viewed with a thrid party tool (e.g. [Paraview](https://www.paraview.org/)).
- `homo_mesh.hpp` Provides a templated mesh container `HomoMesh<Element_t>` for a mesh consisting of a single element type. A voxel element type is defined in `vtkVoxel.hpp` and can be replicated to extend functionality to other element types. For now only meshes in 3D can be saved to a file. If a 2D `Element_t` is needed, it still must use `Point<3,double>` as its point/node/vertex type to maintain compatability with saving to a `.vtk` file.
> This class will likely be significantly changed in the future.
- `Q1.hpp` defines a class `VoxelMeshQ1`, which is essentially `HomoMesh<Voxel>` (where `Voxel` is defined in `vtkVoxel.hpp`) with a constructor to uniformly mesh a box region.


### `gv::fem`
This is where most of the finite element methods are implemented. The handling of sparse matrices and solving of linear systems is primarily handled by [Eigen](https://eigen.tuxfamily.org/dox/index.html) and must be installed somewhere in your system. Header files in `cgeovox` will assume that files from Eigen can be included by (for example) `#include <Eigen/SparseCore>` so the appropriate include path must be given to the compiler/linker (e.g. `-I/usr/include/eigen3` if Eigen was installed using `apt`).


### `gv::pde`
This is where methods for solving specific partial differential equations (e.g., the Poisson equation) are implemented.

### `gv::geometry`
This is where most of the logic that handles (unmeshed) geometry is. For example, reading files generated from [Discrete Element Method](https://en.wikipedia.org/wiki/Discrete_element_method), and putting the particles into an octree so that points in space can be determined to be either inside or outside a particle without looping over every particle. The `gv::geometry::Assembly<Particle_t, n_data>` container is used for this.
- `assembly.hpp` Provides the class `Assembly<Particle_t, n_data>` which uses an octree `ParticleOctree<Particle_t, n_data>`, which is also prvided here.
- `particles.hpp` Provides several particle types that can be used. To use the existing `ParticleOctree<Particle_t, n_data>` class, `Particle_t` must have a method `support(Point<3,double>)` which returns the point on the surface of the particle that is farthest in the given direction (i.e., the point of tangency between the [supporting hyperplane](https://en.wikipedia.org/wiki/Supporting_hyperplane) and the particle). If non-convex particles are used, then a new `ParticleOctree` class would need to be used as the GJK algorithm would no longer be valid.

### `gv::solvers`
This currently contains older code that is unused, but may be useful in the future. If specialized solvers are required and not provided by Eigen (e.g., [Semismooth Newton Method](https://epubs.siam.org/doi/book/10.1137/1.9781611970692) for constrained PDEs), they will be implemented here.




















