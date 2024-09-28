#include "util_module.hpp"
#include "geometry_module.hpp"
#include "mesh_module.hpp"
#include <iostream>


using namespace GeoVox;
using Assembly = geometry::Assembly;
using Mesh = mesh::Mesh;



int test_assembly(){
	Assembly A = Assembly("particles_1.txt");

	// A.print(std::cout);

	std::cout << "BUILDING TREE\n";
	A.make_tree(6);
	// A.print_tree(std::cout);

	std::cout << "SAVING GEOMETRY\n";
	int N[3] {50, 50, 50};
	A.save_geometry("Geometry.dat", 0.875*A._box, N);

	std::cout << "MAKING MIXED MESH\n";
	Mesh M = A.make_mixed_mesh();
	M.saveas("mixed_mesh_from_mesh.vtk");

	std::cout << "MAKING VOXEL MESH\n";
	M = A.make_voxel_mesh();
	M.saveas("voxel_mesh_from_mesh.vtk");
	
	return 1;
}

int main(int argc, char* argv[]){
	// int flag = test_collision();
	int flag = test_assembly();
	return flag;
}