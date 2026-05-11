#include "mesh/voxel_mesh.hpp"
#include "fem/problems/poisson.hpp"
#include "util/sparsity_structure.hpp"

using Mesh_t  = GV::VoxelMesh<10>;
using Poisson = GV::Poisson<Mesh_t,1>;
using DOF_t   = typename Poisson::DOF_t;

int main(int argc, char* argv[]) {
	GV::LogTime t0{"Program"};

	Poisson problem({-1,-1,-1}, {1,1,1});
	
	auto bc_fun1  = [](DOF_t dof) {return 0.0;};
	auto bc_pred1 = [](DOF_t dof) {return dof.key.j()==0;};

	problem.bchandler.add_essential(bc_pred1, bc_fun1);

	problem.set_depth(3);
	// problem.integrate();
	// problem.build_matrices();

	// problem.cache_bc();
	// problem.apply_dirichlet();
	// problem.solve();

	auto predicate = [](DOF_t dof) {return dof.key.x() <= 0.5;};

	for (int r=0; r<3; ++r) {
		GV::LogTime iteration{"Iteration: " + std::to_string(r)};
		if (r>0) {problem.refine(predicate);}
		problem.cache_bc();
		problem.integrate();
		problem.build_matrices();
		problem.apply_dirichlet();
		problem.solve();
	}

	// problem.refine();
	problem.save_as("poissonQ1.vtk");

	// GV::SparseMatImage structure(problem.A);
	// structure.save_as("sparse_mat.bmp");
}