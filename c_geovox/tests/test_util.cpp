#include "util_module.hpp"
#include "geometry_module.hpp"
#include "mesh_module.hpp"
#include <iostream>


using namespace GeoVox;
using Assembly = geometry::Assembly;
using Mesh = mesh::Mesh;
using Point3 = util::Point<3>;
using Box = util::Box;

int test_assembly(){

	std::cout << "READING PARTICLES\n";
	Assembly A = Assembly("particles_1000.txt");

	std::cout << "MAKING PARTICLE OCTREE\n";
	A.divide(7);
	
	// std::cout << "SAVING GEOMETRY\n";
	long unsigned int  N[3] {256, 256, 256};
	// A.save_geometry("Geometry.dat", A.box, N);

	// std::cout << "READING GEOMETRY\n";
	// Point3 H = (A.box.high()-A.box.low())/Point3(N[0], N[1], N[2]);

	// GeoVox::mesh::StructuredPoints SP(Box(A.box.low()+0.5*H, A.box.high()-0.5*H), "Geometry.dat");

	std::cout << "MAKING STRUCTURED POINTS\n";
	GeoVox::mesh::StructuredPoints SP = A.make_structured_mesh(N);
	
	std::cout << "SAVING STRUCTURED POINTS\n";
	SP.saveas("structured_points.vtk");

	std::cout << "MAKING OCTREE STRUCTURE VTK MESH\n";
	Mesh octree_structure = GeoVox::util::visualize_octree_structure<Assembly, GeoVox::geometry::AssemblyNode, GeoVox::geometry::SuperEllipsoid>(&A);

	std::cout << "SAVING OCTREE STRUCTURE AS VTK MESH\n";
	octree_structure.saveas("octree_structure.vtk");

	// std::cout << "MAKING MIXED MESH\n";
	// Mesh M = A.make_mixed_mesh();
	// M.saveas("mixed_mesh_from_mesh.vtk");

	// std::cout << "MAKING VOXEL MESH\n";
	// Mesh M = A.make_voxel_mesh();
	// M.saveas("voxel_mesh.vtk");
	
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