#include "util_module.hpp"
#include "geometry_module.hpp"
#include "mesh_module.hpp"
#include "mac/mac.hpp"

#include "Eigen/Core"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <omp.h>

#define TAU 6.283185307

using namespace GeoVox;
using Assembly = geometry::Assembly;
using Box = util::Box;


////////////////////// GLOBAL VARIABLES FOR ALL TESTS ///////////////////////
Assembly A;
long unsigned int N[3];
double omega[3];



void pressure_derivatives(bool save_files=false){
	extern Assembly A;
	extern long unsigned int N[3];
	extern double omega[3];

	GeoVox::mac::MacMesh mac_approx(A.box,N,&A);
	GeoVox::mac::MacMesh mac_exact(A.box,N,&A);

	#pragma omp parallel for collapse(3)
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
	mac_approx.p = mac_approx.Ap(mac_approx.p);


	// exact derivatives
	double C = -(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);
	double error_max[4] {0};
	double error_L2[4] {0};

	#pragma omp parallel for collapse(3) shared(error_max) reduction(+:error_L2)
	for (long unsigned int k=0; k<N[2]; k++){
		for (long unsigned int j=0; j<N[1]; j++){
			for (long unsigned int i=0; i<N[0]; i++){
				long unsigned int idx = mac_approx.index(i,j,k);
				
				//laplacian
				Point3 X = mac_approx.p_mask.idx2point(i,j,k);
				mac_exact.p[idx] = C*cos(omega[0]*X[0])*cos(omega[1]*X[1])*cos(omega[2]*X[2]);

				double error_at_idx = mac_approx.p[idx]-mac_exact.p[idx];
				error_max[0] = std::max(error_max[0], fabs(error_at_idx));
				error_L2[0] += error_at_idx*error_at_idx;

				//grad_x
				X = mac_approx.u_mask.idx2point(i,j,k);
				mac_exact.u[idx] = -omega[0]*sin(omega[0]*X[0])*cos(omega[1]*X[1])*cos(omega[2]*X[2]);

				error_at_idx = mac_approx.u[idx]-mac_exact.u[idx];
				error_max[1] = std::max(error_max[1], fabs(error_at_idx));
				error_L2[1] += error_at_idx*error_at_idx;


				//grad_y
				X = mac_approx.v_mask.idx2point(i,j,k);
				mac_exact.v[idx] = -omega[1]*cos(omega[0]*X[0])*sin(omega[1]*X[1])*cos(omega[2]*X[2]);
				
				error_at_idx = mac_approx.v[idx]-mac_exact.v[idx];
				error_max[2] = std::max(error_max[2], fabs(error_at_idx));
				error_L2[2] += error_at_idx*error_at_idx;


				//grad_x
				X = mac_approx.w_mask.idx2point(i,j,k);
				mac_exact.w[idx] = -omega[2]*cos(omega[0]*X[0])*cos(omega[1]*X[1])*sin(omega[2]*X[2]);

				error_at_idx = mac_approx.w[idx]-mac_exact.w[idx];
				error_max[3] = std::max(error_max[3], fabs(error_at_idx));
				error_L2[3] += error_at_idx*error_at_idx;
			}
		}
	}

	for (int i=0; i<4; i++){error_L2[i]=sqrt(error_L2[i])/(N[0]*N[1]*N[2]);}

	// print max error_max and error_L2
	std::cout << std::setprecision(10);
	std::cout << "\nPRESSURE DERIVATIVES:\n";
	std::cout << "\tlaplace_p (p DOFS): " << error_max[0] << " (inf-norm) \t" << error_L2[0] << " (L2-mean)" << std::endl;
	std::cout << "\tp_x (u DOFS):       " << error_max[1] << " (inf-norm) \t" << error_L2[1] << " (L2-mean)" << std::endl;
	std::cout << "\tp_y (v DOFS):       " << error_max[2] << " (inf-norm) \t" << error_L2[2] << " (L2-mean)" << std::endl;
	std::cout << "\tp_z (w DOFS):       " << error_max[3] << " (inf-norm) \t" << error_L2[3] << " (L2-mean)" << std::endl;


	// save approximate and exact derivatives
	if (save_files){
		mac_approx.saveas("outfiles/pressure_derivative_approx.vtk");
		mac_exact.saveas("outfiles/pressure_derivative_exact.vtk");
	}
	

	return;
}


void u_velocity_derivatives(bool save_files=false){
	extern Assembly A;
	extern long unsigned int N[3];
	extern double omega[3];
	
	GeoVox::mac::MacMesh mac_approx(A.box,N,&A);
	GeoVox::mac::MacMesh mac_exact(A.box,N,&A);

	//
	#pragma omp parallel for collapse(3)
	for (long unsigned int k=0; k<N[2]; k++){
		for (long unsigned int j=0; j<N[1]; j++){
			for (long unsigned int i=0; i<N[0]; i++){
				long unsigned int idx = mac_approx.index(i,j,k);
				Point3 X = mac_approx.u_mask.idx2point(i,j,k);

				// std::cout << Point3(i,j,k) << "\tpdof= " << mac_approx.p_mask.idx2point(i,j,k) << std::endl;
				// std::cout << Point3(i,j,k) << "\tudof= " << mac_approx.u_mask.idx2point(i,j,k) << std::endl;
				// std::cout << Point3(i,j,k) << "\tvdof= " << mac_approx.v_mask.idx2point(i,j,k) << std::endl;
				// std::cout << Point3(i,j,k) << "\twdof= " << mac_approx.w_mask.idx2point(i,j,k) << std::endl;

				mac_approx.u[idx] = cos(omega[0]*X[0])*cos(omega[1]*X[1])*cos(omega[2]*X[2]);
			}
		}
	}


	// approximate first derivatives
	mac_approx.p = mac_approx.dUdX(mac_approx.u);

	// approximate second derivative
	mac_approx.u = mac_approx.A(mac_approx.u);


	// exact derivatives
	double C = omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]; //negative laplace
	double error_max[2] {0};
	double error_L2[2] {0};

	#pragma omp parallel for collapse(3) shared(error_max) reduction(+:error_L2)
	for (long unsigned int k=0; k<N[2]; k++){
		for (long unsigned int j=0; j<N[1]; j++){
			for (long unsigned int i=0; i<N[0]; i++){
				long unsigned int idx = mac_approx.index(i,j,k);
				
				//laplacian
				Point3 X = mac_approx.u_mask.idx2point(i,j,k);
				mac_exact.u[idx] = C*cos(omega[0]*X[0])*cos(omega[1]*X[1])*cos(omega[2]*X[2]);

				double error_at_idx = mac_approx.u[idx]-mac_exact.u[idx];
				error_max[0] = std::max(error_max[0], fabs(error_at_idx));
				error_L2[0] += error_at_idx*error_at_idx;


				//grad_x
				X = mac_approx.p_mask.idx2point(i,j,k);
				mac_exact.p[idx] = -omega[0]*sin(omega[0]*X[0])*cos(omega[1]*X[1])*cos(omega[2]*X[2]);

				error_at_idx = mac_approx.p[idx]-mac_exact.p[idx];
				error_max[1] = std::max(error_max[1], fabs(error_at_idx));
				error_L2[1] += error_at_idx*error_at_idx;
			}
		}
	}

	for (int i=0; i<1; i++){error_L2[i]=sqrt(error_L2[i])/(N[0]*N[1]*N[2]);}


	// print max errors
	std::cout << std::setprecision(10);
	std::cout << "\nU-VELOCITY DERIVATIVES:\n";
	std::cout << "\tlaplace_u (u DOFS): " << error_max[0] << " (inf-norm) \t" << error_L2[0] << " (L2-mean)" << std::endl;
	std::cout << "\tu_x (p DOFS):       " << error_max[1] << " (inf-norm) \t" << error_L2[1] << " (L2-mean)" << std::endl;

	// save approximate and exact derivatives
	if (save_files){
		mac_approx.saveas("outfiles/u_derivative_approx.vtk");
		mac_exact.saveas("outfiles/u_derivative_exact.vtk");
	}

	return;
}



int main(int argc, char* argv[]){
	//get commandline parameters
	bool save_files = false;
	if (argc > 1){
		save_files = atoi(argv[1]);
	}

	// define geometry and test function parameters
	extern Assembly A;
	A.box = Box(Point3(0,0,0), Point3(1,2,3));

	extern long unsigned int N[3];
	N[0] = 64; 
	N[1] = 64; 
	N[2] = 64;
	
	extern double omega[3];
	omega[0] = 1*TAU/A.box.sidelength()[0];
	omega[1] = 1*TAU/A.box.sidelength()[1];
	omega[2] = 1*TAU/A.box.sidelength()[2];





	// print to consol
	std::cout << "\n========== BEGIN TESTING MAC ==========\n";
	std::cout << "DOMAIN: " << A.box[0] << " to " << A.box[7] << std::endl;
	std::cout << "N: " << N[0] << " " << N[1] << " " << N[2] << std::endl;
	std::cout << "DERIVATIVE_TEST_FUNCTION: f(x,y,z) = cos(" << omega[0] << "x) * cos(" << omega[1] << "y) * cos(" << omega[2] << "z)\n";

	pressure_derivatives(save_files);
	u_velocity_derivatives(save_files);


	std::cout << "\n========== END TESTING MAC ==========\n";
}