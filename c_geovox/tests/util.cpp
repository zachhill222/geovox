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
	Assembly A = Assembly("testdata/sphere.txt");

	std::cout << "MAKING PARTICLE OCTREE\n";
	A.divide(5);

	std::cout << "MAKING OCTREE STRUCTURE VTK MESH\n";
	Mesh octree_structure = GeoVox::util::visualize_octree_structure<Assembly, GeoVox::geometry::AssemblyNode, GeoVox::geometry::SuperEllipsoid>(&A);

	std::cout << "SAVING OCTREE STRUCTURE AS VTK MESH\n";
	octree_structure.saveas("outfiles/octree_structure.vtk");
	

	Box geobox = 2*A.box;
	long unsigned int  N[3] {64, 64, 64};


	// std::cout << "SAVING GEOMETRY\n";
	// A.save_geometry("outfiles/Geometry.dat", A.box, N);

	// std::cout << "READING GEOMETRY\n";
	// Point3 H = (A.box.high()-A.box.low())/Point3(N[0], N[1], N[2]);

	// std::cout << "MAKING STRUCTURED POINTS\n";
	// GeoVox::mesh::StructuredPoints SP = A.make_structured_mesh(geobox,N);
	
	// std::cout << "SAVING STRUCTURED POINTS\n";
	// SP.saveas("outfiles/structured_points.vtk");

	std::cout << "SETTING UP MAC\n";
	GeoVox::mac::MacMesh mac(geobox, N, &A);
	// GeoVox::mac::MacMesh mac(geobox, N);
	mac.f1 = 1*Eigen::VectorXd::Ones(mac.u.size());
	// mac.f2 = 1*Eigen::VectorXd::Ones(mac.v.size());

	mac.mu = 1E-3;
	std::cout << "SOLVING MAC\n";
	// mac.solve_multigrid(50);
	mac.solve(100);
	mac.solve_reverse(100);

	// long unsigned int M[3] {N[0]*2,N[1]*2,N[2]*2};
	// GeoVox::mac::MacMesh test_mac(geobox, M, &A);
	// for (long unsigned int k=0; k<test_mac.N[2]; k++){
	// 	for (long unsigned int j=0; j<test_mac.N[1]; j++){
	// 		for (long unsigned int i=0; i<test_mac.N[0]; i++){
	// 			test_mac.p[test_mac.index(i,j,k)] = mac.fine_index(i,j,k);
	// 		}
	// 	}
	// }

	std::cout << "SAVING MAC SOLUTION\n";
	mac.saveas("outfiles/mac_solution_test.vtk");
	// mac.saveas("outfiles/mac_solution.vtk");
	// mac.saveas("outfiles/mac_solution_multigrid.vtk");

	return 1;
}



int main(int argc, char* argv[]){
	// int flag = test_collision();
	int flag = test_assembly();
	return flag;
}