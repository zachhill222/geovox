#include "fem/problems/block_stokes.hpp"


int main(int argc, char* argv[]) {
	GV::BlockStokes<3> stokes({-1.0,-1.0,-0.25}, {1.0,1.0,0.25});

	using V_DOF_t = typename decltype(stokes)::V_DOF_t;
	stokes.add_velocity_bc([](V_DOF_t dof){return dof.key.z()==0.0 or dof.key.z()==1.0;});
	stokes.set_body_force(1,0,0);

	stokes.set_depth(3);
	stokes.solve_gmres(50);
	stokes.save_as("block_stokes.vtk");

	return 0;
}