#include "util_module.hpp"
#include "geometry_module.hpp"
#include "mesh_module.hpp"
#include <iostream>


using namespace GeoVox;
using Assembly = geometry::Assembly;
using Mesh = mesh::Mesh;


int test_assembly(){
	Assembly A = Assembly("particles_1000.txt");
	A.divide(4);
	
	std::cout << "SAVING GEOMETRY\n";
	int N[3] {200, 200, 200};
	A.save_geometry("Geometry.dat", A.box, N);

	std::cout << "READING GEOMETRY\n";
	GeoVox::mesh::StructuredPoints SP(A.box, "Geometry.dat");

	std::cout << "SAVING STRUCTURED POINTS\n";
	SP.saveas("structured_points.vtk");


	// std::cout << "MAKING MIXED MESH\n";
	// Mesh M = A.make_mixed_mesh();
	// M.saveas("mixed_mesh_from_mesh.vtk");

	std::cout << "MAKING VOXEL MESH\n";
	Mesh M = A.make_voxel_mesh();
	M.saveas("voxel_mesh.vtk");
	
	// std::cout << "GRADIATING\n";
	// A.gradiate();
	// M = A.make_voxel_mesh();
	// M.saveas("voxel_gradiated_mesh.vtk");

	// std::cout << "MAKING SURFACE MESH\n"; //SURFACE MESH IS SLOW
	// Mesh S = M.extract_boundary_mesh();
	// S.saveas("surface_mesh.vtk");

	return 1;
}

int main(int argc, char* argv[]){
	// int flag = test_collision();
	int flag = test_assembly();
	return flag;
}