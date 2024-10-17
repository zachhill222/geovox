#ifndef MAC_H
#define MAC_H


#include "mesh/vtk_structured.hpp"
#include "geometry/assembly.hpp"
#include "util/point.hpp"
#include "util/box.hpp"

#include "Eigen/Core"

#include <string>
#include <sstream>
#include <fstream>
#include <functional> //for passing method functions to solvers

#include <omp.h>

//Convenient typedefs
using StructuredPoints = GeoVox::mesh::StructuredPoints;
using Assembly = GeoVox::geometry::Assembly;
using VectorXd = Eigen::VectorXd;

//Default values
#define MAC_DOMAIN_MARKER 0
#define MAC_DEFAULT_TOL 1E-6
#define MAC_DEFAULT_MAX_OUTER_ITERATIONS 100

namespace GeoVox::mac{
	class MacMesh{
	public:
		MacMesh() : mu(1.0) {}
		
		MacMesh(const Box& box, const long unsigned int N[3], const Assembly& assembly) : mu(1.0), tol(MAC_DEFAULT_TOL), N(N) {
			//get spacing between DOFs
			H = GeoVox::util::div(box.sidelength(),Point3(N[0], N[1], N[2]));

			//create masks
			Point3 offset = 0.5*H;
			Box subbox = Box(box.low()+offset, box.high()-offset);

			p_mask = assembly.make_structured_mesh(subbox, N);

			offset = Point3(0.5*H[0], 0, 0);
			u_mask = assembly.make_structured_mesh(subbox-offset, N);

			offset = Point3(0, 0.5*H[1], 0);
			v_mask = assembly.make_structured_mesh(subbox-offset, N);

			offset = Point3(0, 0, 0.5*H[2]);
			w_mask = assembly.make_structured_mesh(subbox-offset, N);

			//initialize unkowns to 0
			p = VectorXd::Zero(p_mask.N.prod());
			u = VectorXd::Zero(u_mask.N.prod());
			v = VectorXd::Zero(v_mask.N.prod());
			w = VectorXd::Zero(w_mask.N.prod());

			//initialize forcing terms to 0
			f1 = VectorXd::Zero(u.size());
			f2 = VectorXd::Zero(v.size());
			f3 = VectorXd::Zero(w.size());
			g  = VectorXd::Zero(p.size());
		}

		MacMesh(const Box& box, const long unsigned int N[3]) : mu(1.0), tol(MAC_DEFAULT_TOL), N(N) {
			//get spacing between DOFs
			H = GeoVox::util::div(box.sidelength(),Point3(N[0], N[1], N[2]));

			//create masks
			Point3 offset = 0.5*H;
			Box subbox = Box(box.low()+offset, box.high()-offset);

			p_mask = StructuredPoints(subbox, N);

			offset = Point3(0.5*H[0], 0, 0);
			u_mask = StructuredPoints(subbox-offset, N);

			offset = Point3(0, 0.5*H[1], 0);
			v_mask = StructuredPoints(subbox-offset, N);

			offset = Point3(0, 0, 0.5*H[2]);
			w_mask = StructuredPoints(subbox-offset, N);

			//initialize unkowns to 0
			p = VectorXd::Zero(p_mask.N.prod());
			u = VectorXd::Zero(u_mask.N.prod());
			v = VectorXd::Zero(v_mask.N.prod());
			w = VectorXd::Zero(w_mask.N.prod());

			//initialize forcing terms to 0
			f1 = VectorXd::Zero(u.size());
			f2 = VectorXd::Zero(v.size());
			f3 = VectorXd::Zero(w.size());
			g  = VectorXd::Zero(p.size());
		}

		//Viscosity
		double mu;

		//Solver parameters
		int max_outer_iterations; //Maximum number of outer Distributive Gauss-Seidel (DGS) iterations
		double tol; //Outer iteration residual tolerance for each equation (u,v,w) momentum equations (p) mass conservation equation
		// double SCALE; //Scaling factor to multiply both sides. Should be O(h^2).

		//DOF masks (periodic BC, so the number of DOFs are the same)
		Eigen::Matrix<long unsigned int, 3, 1> N;
		Point3 H;
		inline long unsigned int index(long unsigned int i, long unsigned int j, long unsigned k) const {return p_mask.index(i%N[0],j%N[1],k%N[2]);}

		StructuredPoints p_mask; //pressure mask (global indexing): Nx by Ny by Nz
		StructuredPoints u_mask; //x-velocity mask (global indexing): Nx by Ny by Nz
		StructuredPoints v_mask; //y-velocity mask (global indexing): Nx by Ny by Nz
		StructuredPoints w_mask; //z-velocity mask (global indexing): Nx by Ny by Nz

		//DOF values
		VectorXd p; //pressure values (reduced indexing): number of nonzeros in p_mask
		VectorXd u; //x-velocity values (reduced indexing): number of nonzeros in
		VectorXd v; //y-velocity values (reduced indexing)
		VectorXd w; //z-velocity values (reduced indexing)

		//RHS values
		VectorXd f1; //forcing term in x-direction at x-DOFs
		VectorXd f2; //forcing term in y-direction at y-DOFs
		VectorXd f3; //forcing term in z-direction at z-DOFsS
		VectorXd g;  //forcing term for p at p-DOFs
		
		//FIRST DERIVATIVES
		VectorXd dPdX(const VectorXd& variable) const;
		VectorXd dPdY(const VectorXd& variable) const;
		VectorXd dPdZ(const VectorXd& variable) const;

		VectorXd dUdX(const VectorXd& variable) const;
		VectorXd dVdY(const VectorXd& variable) const;
		VectorXd dWdZ(const VectorXd& variable) const;


		//SECOND DERIVATIVES
		VectorXd A(const VectorXd& variable) const; //same template for u,v,w
		VectorXd Ap(const VectorXd& variable) const;

		//SOLUTION
		void GS_relax_velocity();
		VectorXd GS_relax_p() const; //Return ep = Ap_inv(g-Bx(u)-By(v)-Bz(w))
		void DGS();
		void solve(const int max_iter=MAC_DEFAULT_MAX_OUTER_ITERATIONS);



		//save solution interpolated to pressure DOFs
		void saveas(const std::string filename) const;
	};
}


#endif