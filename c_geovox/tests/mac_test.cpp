#include "util_module.hpp"
#include "geometry_module.hpp"
#include "mesh_module.hpp"
#include "mac/mac.hpp"

#include "Eigen/Core"

#include <iostream>
#include <vector>
#include <cmath>

#include <omp.h>

#define TAU 6.283185307

using namespace GeoVox;
using Assembly = geometry::Assembly;
using Box = util::Box;

void mac_pressure_derivatives(){
	// set up geometry
	Assembly A;
	A.box = Box(Point3(0,0,0), TAU*Point3(1,1,1));

	long unsigned int N[3] {128,128,128};
	GeoVox::mac::MacMesh mac_approx(A.box,N,&A);
	GeoVox::mac::MacMesh mac_exact(A.box,N,&A);


	// define pressure
	Point3 H = A.box.sidelength();
	double omega[3] {1*TAU/H[0], 1*TAU/H[1], 1*TAU/H[2]}; //frequency

	for (long unsigned int k=0; k<N[2]; k++){
		for (long unsigned int j=0; j<N[1]; j++){
			for (long unsigned int i=0; i<N[0]; i++){
				long unsigned int idx = mac_approx.index(i,j,k);
				Point3 X = mac_approx.p_mask.idx2point(i,j,k);

				// std::cout << Point3(i,j,k) << "\tpdof= " << mac_approx.p_mask.idx2point(i,j,k) << std::endl;
				// std::cout << Point3(i,j,k) << "\tudof= " << mac_approx.u_mask.idx2point(i,j,k) << std::endl;
				// std::cout << Point3(i,j,k) << "\tvdof= " << mac_approx.v_mask.idx2point(i,j,k) << std::endl;
				// std::cout << Point3(i,j,k) << "\twdof= " << mac_approx.w_mask.idx2point(i,j,k) << std::endl;

				mac_approx.p[idx] = cos(omega[0]*X[0])*cos(omega[1]*X[1])*cos(omega[2]*X[2]);
			}
		}
	}


	// approximate first derivatives
	mac_approx.u = mac_approx.dPdX(mac_approx.p);
	mac_approx.v = mac_approx.dPdY(mac_approx.p);
	mac_approx.w = mac_approx.dPdZ(mac_approx.p);

	// approximate second derivative
	// mac_approx.p = mac_approx.Ap(mac_approx.p);


	// exact derivatives
	double C = omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]; //negative laplace
	double error[4] {0};

	for (long unsigned int k=0; k<N[2]; k++){
		for (long unsigned int j=0; j<N[1]; j++){
			for (long unsigned int i=0; i<N[0]; i++){
				long unsigned int idx = mac_approx.index(i,j,k);
				
				//laplacian
				Point3 X = mac_approx.p_mask.idx2point(i,j,k);
				mac_exact.p[idx] = C*cos(omega[0]*X[0])*cos(omega[1]*X[1])*cos(omega[2]*X[2]);
				error[0] = std::max(error[0], fabs(mac_approx.p[idx]-mac_exact.p[idx]));


				//grad_x
				X = mac_approx.u_mask.idx2point(i,j,k);
				mac_exact.u[idx] = -omega[0]*sin(omega[0]*X[0])*cos(omega[1]*X[1])*cos(omega[2]*X[2]);
				error[1] = std::max(error[1], fabs(mac_approx.u[idx]-mac_exact.u[idx]));


				//grad_y
				X = mac_approx.v_mask.idx2point(i,j,k);
				mac_exact.v[idx] = -omega[1]*cos(omega[0]*X[0])*sin(omega[1]*X[1])*cos(omega[2]*X[2]);
				error[2] = std::max(error[2], fabs(mac_approx.v[idx]-mac_exact.v[idx]));


				//grad_x
				X = mac_approx.w_mask.idx2point(i,j,k);
				mac_exact.w[idx] = -omega[2]*cos(omega[0]*X[0])*cos(omega[1]*X[1])*sin(omega[2]*X[2]);
				error[3] = std::max(error[3], fabs(mac_approx.w[idx]-mac_exact.w[idx]));
			}
		}
	}


	// print max errors
	std::cout << "TESTING PRESSURE DERIVATIVES\n";
	std::cout << "\tDOMAIN: " << A.box[0] << " to " << A.box[7] << std::endl;
	std::cout << "\tN: " << N[0] << " " << N[1] << " " << N[2] << std::endl;
	std::cout << "\tMAX_ERROR: u = cos(" << omega[0] << "x) * cos(" << omega[1] << "y) * cos(" << omega[2] << "z)\n";
	std::cout << "\t\tlaplace_p (p DOFS): " << error[0] << std::endl;
	std::cout << "\t\tp_x (u DOFS): " << error[1] << std::endl;
	std::cout << "\t\tp_y (v DOFS): " << error[2] << std::endl;
	std::cout << "\t\tp_z (w DOFS): " << error[3] << std::endl;

	// save approximate and exact derivatives
	mac_approx.saveas("outfiles/pressure_derivative_approx.vtk");
	mac_exact.saveas("outfiles/pressure_derivative_exact.vtk");

	return;
}



int main(int argc, char* argv[]){
	mac_pressure_derivatives();
}