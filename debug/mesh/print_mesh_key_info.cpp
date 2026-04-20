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
	Point_t low{0,0,0};
	Point_t high{1,1,1};
	Mesh_t  mesh(low, high);

	//print all elements at a depth
	auto print_elem = [](Elem_t el) {
		std::cout << "=== Element " << el.linear_index() << "\n"
				  << "dijk: (" << el.depth() << ", " << el.i() << ", " << el.j() << ", " << el.k() << ")\n"
				  << "color: " << el.color() << "\n"
		          << "depth_linear: " << el.depth_linear_index() << "\n";
		std::cout << "verts (global): ";
		for (int v=0; v<8; ++v) {std::cout << el.vertex(v).linear_index() << " ";}
		std::cout << "\nverts (reduced): ";
		for (int v=0; v<8; ++v) {std::cout << el.vertex(v).reduced_key().linear_index() << " ";}
		std::cout << "\n\n";
	};

	//print all vertices at a depth
	auto print_vert = [](Vert_t vtx) {
		std::cout << "=== Vertex " << vtx.linear_index() << "\n"
				  << "dijk: (" << vtx.depth() << ", " << vtx.i() << ", " << vtx.j() << ", " << vtx.k() << ")\n"
				  << "color: " << vtx.color() << "\n"
		          << "depth_linear: " << vtx.depth_linear_index() << "\n";

		Vert_t red = vtx.reduced_key();
		std::cout << "reduced index: " << red.linear_index() << "\n"
		          << "reduced dijk: (" << red.depth() << ", " << red.i() << ", " << red.j() << ", " << red.k() << ")\n";
		std::cout << "\n\n";
	};
	
	mesh.for_each_depth<Elem_t>(d,print_elem);
	mesh.for_each_depth<Vert_t>(d,print_vert);

	
	return 0;
}