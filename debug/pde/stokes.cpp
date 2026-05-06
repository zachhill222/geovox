#include "fem/problems/stokes.hpp"
#include "util/point.hpp"
#include "util/compatibility.hpp"

using Stokes = GV::Stokes<1,8>;

int main(int argc, char* argv[])
{
	Stokes stokes;
	stokes.set_depth(1);

	using DOF_t = typename Stokes::DOF_t;
	
	// stokes.add_velocity_bc([](DOF_t dof){return dof.key.x()==0.0;}, [](DOF_t dof){return GV::Point<3,double>{1,0,0};});
	// stokes.add_velocity_bc([](DOF_t dof){
	// 	return 	dof.key.y()==0.0 or dof.key.y()==1.0 or
	// 			dof.key.z()==0.0 or dof.key.z()==1.0;});

	// std::vector<double> F(stokes.U.size(), 0.0);
	// for (size_t i=0; i<stokes.U.size()/2; ++i) {F[i]=0.1;}

	// std::vector<double> H(stokes.P.size(), 0.0);

	// for (int m=0; m<40; ++m) {
	// 	stokes.standard_uzawa<true>(GV::as_span(stokes.U), GV::as_span(stokes.P), GV::as_span(F), GV::as_span(H), 0.5, 1);
	// 	stokes.standard_uzawa<false>(GV::as_span(stokes.U), GV::as_span(stokes.P), GV::as_span(F), GV::as_span(H), 0.5, 1);
	// }

	stokes.save_as("stokes.vtk");
	return 0;
}