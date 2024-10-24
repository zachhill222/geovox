#include "util_module.hpp"
#include "geometry_module.hpp"
#include "mesh_module.hpp"
#include "mac/mac.hpp"

#include "Eigen/Core"

#include <iostream>
#include <vector>
#include <cmath>

#include <omp.h>

using namespace GeoVox;
using Assembly = geometry::Assembly;
using Mesh = mesh::Mesh;
using Point3 = util::Point<3>;
using Box = util::Box;

int test_assembly(){
	std::cout << "READING PARTICLES\n";
	Assembly A = Assembly("particles_50.txt");

	std::cout << "MAKING PARTICLE OCTREE\n";
	A.divide(5);

	std::cout << "MAKING OCTREE STRUCTURE VTK MESH\n";
	Mesh octree_structure = GeoVox::util::visualize_octree_structure<Assembly, GeoVox::geometry::AssemblyNode, GeoVox::geometry::SuperEllipsoid>(&A);

	std::cout << "SAVING OCTREE STRUCTURE AS VTK MESH\n";
	octree_structure.saveas("octree_structure.vtk");
	

	Box geobox = 1.05*A.box;
	long unsigned int  N[3] {128, 128, 128};


	// std::cout << "SAVING GEOMETRY\n";
	// A.save_geometry("Geometry.dat", A.box, N);

	// std::cout << "READING GEOMETRY\n";
	// Point3 H = (A.box.high()-A.box.low())/Point3(N[0], N[1], N[2]);

	// std::cout << "MAKING STRUCTURED POINTS\n";
	// GeoVox::mesh::StructuredPoints SP = A.make_structured_mesh(geobox,N);
	
	// std::cout << "SAVING STRUCTURED POINTS\n";
	// SP.saveas("structured_points.vtk");

	std::cout << "SETTING UP MAC\n";
	GeoVox::mac::MacMesh mac(geobox, N, &A);
	// GeoVox::mac::MacMesh mac(geobox, N);
	mac.f1 = 1*Eigen::VectorXd::Ones(mac.u.size());
	// mac.f2 = 1*Eigen::VectorXd::Ones(mac.v.size());

	mac.mu = 1E-3;
	std::cout << "SOLVING MAC\n";
	mac.solve(101);

	std::cout << "SAVING MAC SOLUTION\n";
	mac.saveas("mac_solution.vtk");

	return 1;
}



int main(int argc, char* argv[]){
	// int flag = test_collision();
	int flag = test_assembly();
	return flag;
}