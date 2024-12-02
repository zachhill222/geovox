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

void mark_regions(std::string filename, long unsigned int N[3]){
	Assembly A(filename);
	A.divide(5);
	
	VoxelParticleGeometry voxel_mesh(A, N);

	// voxel_mesh.periodicBC[0] = false;
	// voxel_mesh.periodicBC[1] = false;
	// voxel_mesh.periodicBC[2] = false;

	voxel_mesh.compute_connectivity();
	voxel_mesh.saveas("outfiles/mesh/voxel_mesh_connectivity.vtk");
}



int main(int argc, char* argv[]){
	//get commandline parameters
	long unsigned int n = 16;
	if (argc > 1){
		n = atoi(argv[1]);
	}
	long unsigned int N[3] {n, n, n};

	std::string filename = "testdata/sphere.txt";
	if (argc > 2){
		filename = argv[2];
	}

	// print to consol
	std::cout << "\n========== BEGIN TESTING MESH ==========\n";
	std::cout << "ASSEMBLY: " << filename << std::endl;
	std::cout << "N: " << N[0] << " " << N[1] << " " << N[2] << std::endl;

	mark_regions(filename, N);


	std::cout << "\n========== END TESTING MAC ==========\n";
}