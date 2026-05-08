#include "mesh/voxel_mesh.hpp"
#include "fem/problems/poisson.hpp"
#include "util/sparsity_structure.hpp"

using Mesh_t  = GV::VoxelMesh<6>;
using Poisson = GV::Poisson<Mesh_t,6>;
using DOF_t   = typename Poisson::DOF_t;

int main(int argc, char* argv[]) {
	GV::LogTime t0{"Program"};

	Poisson problem({0,0,0}, {1,1,1});
	
	auto bc_fun1  = [](DOF_t dof) {return 0.0;};
	auto bc_pred1 = [](DOF_t dof) {return dof.key.depth_linear_index()==0;};

	// auto bc_fun2  = [](DOF_t dof) {return 0.0;};
	// auto bc_pred2 = [](DOF_t dof) {return dof.key.x()==0.0;};

	problem.bchandler.add_essential(bc_pred1, bc_fun1);
	// problem.bchandler.add_essential(bc_pred2, bc_fun2);

	problem.set_depth(4);
	problem.integrate();
	problem.build_matrices();

	problem.cache_bc();
	problem.apply_dirichlet();
	problem.solve();
	problem.save_as("poissonQ1.vtk");

	GV::SparseMatImage structure(problem.A);
	structure.save_as("sparse_mat.bmp");
}