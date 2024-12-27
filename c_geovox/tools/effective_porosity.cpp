#include "util_module.hpp"
#include "geometry_module.hpp"
#include "mesh_module.hpp"
#include "mac/mac.hpp"

#include "Eigen/Core"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

using namespace GeoVox;
using Assembly = geometry::Assembly;
using Box = util::Box;
using VoxelParticleGeometry = geometry::VoxelParticleGeometry;

void mark_regions(const Assembly& A, long unsigned int N[3], const std::string group_name, const std::string outfolder){
	std::cout << "\n=============== discretization: " << N[0] << " " << N[1] << " " << N[2] << " ===============\n";

	VoxelParticleGeometry voxel_mesh = A.make_structured_mesh(N);

	voxel_mesh.periodic_bc[0] = false;
	voxel_mesh.periodic_bc[1] = false;
	voxel_mesh.periodic_bc[2] = false;

	voxel_mesh.wall_bc[0] = false;
	voxel_mesh.wall_bc[1] = true;
	voxel_mesh.wall_bc[2] = true;
	voxel_mesh.wall_bc[3] = true;
	voxel_mesh.wall_bc[4] = true;
	voxel_mesh.wall_bc[5] = true;

	voxel_mesh.compute_connectivity();
	voxel_mesh.print(std::cout);


	std::string outfilename = outfolder + "/" + group_name + "_" + std::to_string(N[0]) + "x" + std::to_string(N[1]) + "x" + std::to_string(N[2]) + ".vtk";
	std::cout << "saving discretization as: " << outfilename << std::endl;
	voxel_mesh.saveas(outfilename);
}


int main(int argc, char* argv[]){
	//SET FILE
	std::string filename = argv[1];
	std::cout << "FILE: " << filename << std::endl;

	//SET OUTFOLDER
	std::string outfolder;
	if (argc<4){
		outfolder = "outfiles";
	}else{
		outfolder = argv[3];
	}
	std::cout << "OUTFOLDER: " << outfolder << std::endl;

	
	//GET GROUP NAME
	int start_idx = filename.size();
	for (start_idx=filename.size(); start_idx>=1; start_idx--){
		if (filename[start_idx-1]=='/'){break;}
	}

	int end_idx = filename.size();
	for (end_idx=filename.size()-1; end_idx>=0; end_idx--){
		if (filename[end_idx]=='.'){break;}
	}

	std::string group_name = "";
	for (int i=start_idx; i<end_idx; i++){
		group_name += filename[i];
	}


	//READ ASSEMBLY
	Assembly assembly(filename, argv[2]);

	//PREPARE ASSEMBLY
	assembly.divide(5);
	std::cout << assembly.tostr() << std::endl;

	//CHECK CONNECTIVITY AND POROSITY
	std::vector<long unsigned int> discretization {128, 150, 200, 256, 512};
	for (long unsigned int idx=0; idx<discretization.size(); idx++){
		long unsigned int N[3] {discretization[idx], discretization[idx], discretization[idx]};
		mark_regions(assembly, N, group_name, outfolder);
	}
}