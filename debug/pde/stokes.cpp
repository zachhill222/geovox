#include "fem/problems/stokes.hpp"
#include "util/point.hpp"
#include "util/compatibility.hpp"
#include "util/log_time.hpp"

using Stokes = GV::Stokes<1,0,8>;

int main(int argc, char* argv[])
{
	GV::LogTime timer{"Program Time"};

	int refine_depth = 3;
	if (argc>1) {refine_depth = atoi(argv[1]);}


	Stokes stokes(1,1,1);
	stokes.set_depth(1);
	stokes.set_body_force(100.0,0.0,0.0);

	// stokes.check_stokes_op();

	using DOF_t = typename Stokes::V_DOF_t;
	
	// stokes.add_velocity_bc([](DOF_t dof){return dof.key.x()==0.0;}, [](DOF_t dof){return GV::Point<3,double>{1,0,0};});
	stokes.add_velocity_bc([](DOF_t dof){
		return 	dof.key.y()==0.0 or dof.key.y()==1.0 or
				dof.key.z()==0.0 or dof.key.z()==1.0;});
	
	std::vector<double> rhs;
	for (int r=0; r<refine_depth; ++r) {
		if (r>0) {stokes.refine();}

		stokes.cache_bc();

		rhs.assign(stokes.X.size(), 0.0);
		stokes.compute_F(GV::as_span(rhs,0,stokes.n_vel_total()));
		stokes.smooth<1>(5, rhs, 1e-5,true);
		stokes.save_as("stokes_"+std::to_string(r)+".vtk");
	}

	// stokes.smooth(100,rhs,1e-10,true);


	return 0;
}