#include "gutil.hpp"
#include "mesh/voxel_mesh.hpp"


using Point_t = gutil::Point<3,double>;
using Mesh_t  = GV::VoxelMesh<6>;
using Elem_t  = Mesh_t::VoxelElement;
using Vert_t  = Mesh_t::VoxelVertex;

int main(int argc, char* argv[])
{
	//get depth
	size_t d=1;
	if (argc>1) {
		const size_t dd = atoi(argv[1]);
		if (dd>=0 and dd<=Mesh_t::MAX_DEPTH) {
			d = dd;
		}
	}
	
	//build test mesh
	Point_t low{-1,-1,-1};
	Point_t high{1,1,1};
	Mesh_t  mesh(low, high);

	//print all elements at a depth
	auto pred = [&](Elem_t el) {
		auto v1 = el.vertex(0);
		auto v2 = el.vertex(7);
		auto center = 0.5*(mesh.ref2geo(v1)+mesh.ref2geo(v2));
		return gutil::squaredNorm(center) < 0.25;
	};

	mesh.set_depth(1);
	mesh.refine_to_depth(Elem_t{1,0,0,0},d,pred);
	mesh.save_unstructured_mesh("unstructured.vtk");

	return 0;
}