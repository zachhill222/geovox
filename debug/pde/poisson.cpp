#include "fem/problems/poisson.hpp"
#include "mesh/voxel_mesh.hpp"
#include "util/point.hpp"

using Mesh_t = GV::VoxelMesh<6>;
using Poisson_t = GV::Poisson<Mesh_t,1>;
using DOF_t = typename Poisson_t::DOF_t;
using Vec_t = typename Poisson_t::Vec_t;

int main(int argc, char* argv[]) {
	Poisson_t problem{GV::Point<3,double>{-1,-1,-1}, GV::Point<3,double>{1,1,1}};
	problem.bchandler.add_essential([](DOF_t dof){return (dof.key.y()==0) && (dof.key.z()==0);}, [](DOF_t dof){return 0.0;});

	problem.set_depth(5);
	problem.integrate();
	problem.build_matrices();

	Vec_t ones = Vec_t::Ones(problem.rhs.size());
	problem.apply_dirichlet();
	problem.solve();
	problem.save_as("poisson.vtk");
}