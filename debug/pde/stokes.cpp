#include "fem/problems/stokes.hpp"
#include "util/point.hpp"
#include "util/compatibility.hpp"

using Stokes = GV::Stokes<1,0,8>;

int main(int argc, char* argv[])
{
	Stokes stokes;
	stokes.set_depth(0);
	stokes.set_body_force(1.0,0.0,0.0);

	using DOF_t = typename Stokes::V_DOF_t;

	// stokes.add_velocity_bc([](DOF_t dof){return dof.key.x()==0.0;}, [](DOF_t dof){return GV::Point<3,double>{1,0,0};});
	stokes.add_velocity_bc([](DOF_t dof){
		return 	dof.key.y()==0.0 or dof.key.y()==1.0 or
				dof.key.z()==0.0 or dof.key.z()==1.0;});
	
	std::vector<double> F,H;
	for (int r=0; r<4; ++r) {
		if (r>0) {stokes.refine();}

		stokes.cache_bc();

		F.assign(stokes.U.size(), 0.0);
		stokes.compute_F(F);

		H.assign(stokes.P.size(), 0.0);

		for (int m=0; m<50; ++m) {
			stokes.standard_uzawa<true>(GV::as_span(stokes.U), GV::as_span(stokes.P), GV::as_span(F), GV::as_span(H), 0.5, 10);
			stokes.standard_uzawa<false>(GV::as_span(stokes.U), GV::as_span(stokes.P), GV::as_span(F), GV::as_span(H), 0.5, 10);
		}
		stokes.save_as("stokes_"+std::to_string(r)+".vtk");
	}
	return 0;
}