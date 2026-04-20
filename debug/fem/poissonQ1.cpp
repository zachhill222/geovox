#include "mesh/voxel_mesh.hpp"
#include "fem/problems/poisson.hpp"


using Mesh_t  = GV::VoxelMesh<8>;
using Poisson = GV::Poisson<Mesh_t>;
using DOF_t   = typename Poisson::DOF_t;

int main(int argc, char* argv[]) {
	GV::LogTime t0{"Program"};

	Poisson problem({0,0,0}, {1,1,1});
	problem.set_depth(5);
	problem.integrate();
	problem.build_matrices();

	auto bc_fun1  = [](DOF_t dof) {return dof.key.x();};
	auto bc_pred1 = [](DOF_t dof) {return dof.key.y()==1.0;};

	auto bc_fun2  = [](DOF_t dof) {return dof.key.y();};
	auto bc_pred2 = [](DOF_t dof) {return dof.key.x()==1.0;};

	problem.bchandler.add_essential(bc_pred1, bc_fun1);
	problem.bchandler.add_essential(bc_pred2, bc_fun2);
	problem.apply_dirichlet();
	problem.solve();
	problem.save_as("poissonQ1.vtk");
}