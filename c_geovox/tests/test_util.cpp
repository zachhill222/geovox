#include "util_module.hpp"
#include "geometry_module.hpp"
#include "mesh_module.hpp"
#include "mac/mac.hpp"

#include <iostream>
#include <vector>


using namespace GeoVox;
using Assembly = geometry::Assembly;
using Mesh = mesh::Mesh;
using Point3 = util::Point<3>;
using Box = util::Box;

int test_assembly(){
	std::cout << "READING PARTICLES\n";
	Assembly A = Assembly("particles_1.txt");

	std::cout << "MAKING PARTICLE OCTREE\n";
	A.divide(5);
	
	// std::cout << "SAVING GEOMETRY\n";
	long unsigned int  N[3] {50, 50, 50};
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

	std::cout << "SETTING UP MAC\n";
	GeoVox::mac::MacMesh mac(SP.box, SP.N, A);
	mac.f1 = std::vector<double>(mac.u.size(), 1.0);

	// std::cout << "SOLVING MAC\n";
	// for (int i=0; i<10; i++){
	// 	std::cout << "\titeration " << i << std::endl;
	// 	mac.DGS();
	// }

	std::cout << "SAVING MAC SOLUTION\n";
	mac.saveas("mac_solution.vtk");


	return 1;
}



int main(int argc, char* argv[]){
	// int flag = test_collision();
	int flag = test_assembly();
	return flag;
}