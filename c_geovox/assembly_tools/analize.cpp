#include "util_module.hpp"
#include "geometry_module.hpp"
#include "mesh_module.hpp"
#include "mac/mac.hpp"

#include "Eigen/Core"

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

using namespace GeoVox;
using Assembly = geometry::Assembly;
using Box = util::Box;
using VoxelParticleGeometry = geometry::VoxelParticleGeometry;

void mark_regions(const Assembly& A, long unsigned int N[3]){
	VoxelParticleGeometry voxel_mesh(A, N);

	// voxel_mesh.periodic_bc[0] = false;
	// voxel_mesh.periodic_bc[1] = false;
	// voxel_mesh.periodic_bc[2] = false;

	voxel_mesh.compute_connectivity();
	voxel_mesh.saveas("outfiles/voxel_mesh_connectivity.vtk");
	voxel_mesh.print(std::cout);
}


int main(int argc, char* argv[]){
	//SET FILE
	std::string filename = argv[1];
	std::cout << "FILE: " << filename << std::endl;

	//READ ASSEMBLY
	Assembly assembly(filename, argv[2]);

	//PREPARE ASSEMBLY
	assembly.divide(5);

	//CHECK CONNECTIVITY AND POROSITY
	long unsigned int N[3] {128, 128, 128};
	mark_regions(assembly, N);
}