#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <iostream>

#include "gutil.hpp"
#include "geovox.hpp"


using Scalar_t   = double;
using Assembly_t = GV::SignedDistanceSpheres<Scalar_t, 0b111, false, 1>; //last parameter 0 for non-compact support tanh heaviside
using Stokes_t   = GV::DiffuseStokes<Assembly_t, 3>;

int main() {
	Stokes_t::Box_t domain{{-2,-2,-2}, {2,2,2}};
	Stokes_t stokes(domain, 6);

	stokes.add_unit_sphere();
	stokes.body_acceleration = {1, 0, 0};   // drive flow along x
	stokes.viscosity = 1;
	stokes.eps_scale = 2;

	stokes.inner_iter.max_iter = 1000;
	stokes.outer_iter.max_iter = 1000;
	stokes.initialize(2);
	stokes.update_eps();

	// a few interface-refinement passes before solving, matching the
	// established geometry_refine pattern from the scalar convergence test
	for (int i=0; i<4; ++i) {
		stokes.geometry_refine(1);
		// stokes.refine_interior();
		stokes.update_eps();   // min_element_size changes after refinement -- eps needs recomputing each time
		GV::ddm_stokes_monolithic(stokes);
		// GV::ddm_stokes_standard_uzawa(stokes);
		// GV::ddm_stokes_cd_uzawa(stokes);
	}


	stokes.save_as("stokes_result.vtk");

	std::cout << "Done. n_pressure_dofs=" << stokes.p_handler.n_dofs()
	          << " n_velocity_dofs=" << stokes.u_handler.n_dofs() << "\n";
	return 0;
}