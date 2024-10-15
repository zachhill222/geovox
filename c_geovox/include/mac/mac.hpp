#ifndef MAC_H
#define MAC_H

#define MAC_DOMAIN_MARKER 0

#include "mesh/vtk_structured.hpp"
#include "geometry/assembly.hpp"
#include "util/point.hpp"
#include "util/box.hpp"

#include "Eigen/Core"

#include <string>
#include <sstream>
#include <fstream>
#include <functional> //for passing method functions to solvers

using StructuredPoints = GeoVox::mesh::StructuredPoints;
using Assembly = GeoVox::geometry::Assembly;
using VectorXd = Eigen::VectorXd;


namespace GeoVox::mac{
	class MacMesh{
	public:
		MacMesh() : mu(1.0) {}
		
		MacMesh(const Box& box, const long unsigned int N[3], const Assembly& assembly) : mu(1.0) {
			//get spacing between DOFs
			Point3 H = GeoVox::util::div((box.high()-box.low()).eval(),Point3(N[0], N[1], N[2]));
			long unsigned int M[3];

			//create masks
			Point3 offset = 0.5*H;
			Box subbox = Box(box.low()+offset, box.high()-offset);
			p_mask = assembly.make_structured_mesh(subbox, N);

			offset = Point3(0, 0.5*H[1], 0.5*H[2]); 
			subbox = Box(box.low()+offset, box.high()-offset);
			M[0] = N[0]+1; M[1] = N[1]; M[2] = N[2];
			u_mask = assembly.make_structured_mesh(subbox, M);

			offset = Point3(0.5*H[0], 0, 0.5*H[2]); 
			subbox = Box(box.low()+offset, box.high()-offset);
			M[0] = N[0]; M[1] = N[1]+1; M[2] = N[2];
			v_mask = assembly.make_structured_mesh(subbox, M);

			offset = Point3(0.5*H[0], 0.5*H[1], 0); 
			subbox = Box(box.low()+offset, box.high()-offset);
			M[0] = N[0]; M[1] = N[1]; M[2] = N[2]+1;
			w_mask = assembly.make_structured_mesh(subbox, M);

			//initialize unkowns to 0
			p = VectorXd::Zero( N[0]    *  N[1]    *  N[2]    );
			u = VectorXd::Zero((N[0]+1) *  N[1]    *  N[2]    );
			v = VectorXd::Zero( N[0]    * (N[1]+1) *  N[2]    );
			w = VectorXd::Zero( N[0]    *  N[1]    * (N[2]+1) );

			//initialize forcing terms to 0
			f1 = VectorXd::Zero(u.size());
			f2 = VectorXd::Zero(v.size());
			f3 = VectorXd::Zero(w.size());
			g  = VectorXd::Zero(p.size());
		}

		MacMesh(const Box& box, const long unsigned int N[3]){
			//get spacing between DOFs
			Point3 H = GeoVox::util::div((box.high()-box.low()).eval(),Point3(N[0], N[1], N[2]));
			long unsigned int M[3];

			//create masks
			Point3 offset = 0.5*H;
			Box subbox = Box(box.low()+offset, box.high()-offset);

			p_mask = StructuredPoints(subbox, N);

			offset = Point3(0, 0.5*H[1], 0.5*H[2]); 
			subbox = Box(box.low()+offset, box.high()-offset);
			M[0] = N[0]+1; M[1] = N[1]; M[2] = N[2];
			u_mask = StructuredPoints(subbox, M);

			offset = Point3(0.5*H[0], 0, 0.5*H[2]); 
			subbox = Box(box.low()+offset, box.high()-offset);
			M[0] = N[0]; M[1] = N[1]+1; M[2] = N[2];
			v_mask = StructuredPoints(subbox, M);

			offset = Point3(0.5*H[0], 0.5*H[1], 0); 
			subbox = Box(box.low()+offset, box.high()-offset);
			M[0] = N[0]; M[1] = N[1]; M[2] = N[2]+1;
			w_mask = StructuredPoints(subbox, M);

			//initialize unkowns to 0
			p = VectorXd::Zero( N[0]    *  N[1]    *  N[2]   );
			u = VectorXd::Zero((N[0]+1) *  N[1]    *  N[2]   );
			v = VectorXd::Zero( N[0]    * (N[1]+1) *  N[2]   );
			w = VectorXd::Zero( N[0]    *  N[1]    * (N[2]+1));

			//initialize forcing terms to 0
			f1 = VectorXd::Zero(u.size());
			f2 = VectorXd::Zero(v.size());
			f3 = VectorXd::Zero(w.size());
			g  = VectorXd::Zero(p.size());
		}

		//Viscosity
		double mu;

		//DOF masks
		StructuredPoints p_mask; //pressure mask (global indexing): Nx by Ny by Nz
		StructuredPoints u_mask; //x-velocity mask (global indexing): (Nx+1) by Ny by Nz
		StructuredPoints v_mask; //y-velocity mask (global indexing): Nx by (Ny+1) by Nz
		StructuredPoints w_mask; //z-velocity mask (global indexing): Nx by Ny by (Nz+1)

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
		
		//gradients of pressure
		VectorXd discrete_partial_t(const StructuredPoints& eval_dof_mask, const StructuredPoints& given_dof_mask, const VectorXd& vals, const int direction) const;
		inline VectorXd Bx_t(const VectorXd& p) const {return discrete_partial(u_mask, p_mask, p, 0);} //discrete partial_x (row-vector)
		inline VectorXd By_t(const VectorXd& p) const {return discrete_partial(v_mask, p_mask, p, 1);} //discrete partial_y (row-vector)
		inline VectorXd Bz_t(const VectorXd& p) const {return discrete_partial(w_mask, p_mask, p, 2);} //discrete partial_z (row-vector)

		//divergence of velocity
		VectorXd discrete_partial(const StructuredPoints& eval_dof_mask, const StructuredPoints& given_dof_mask, const VectorXd& vals, const int direction) const;
		inline VectorXd Bx(const VectorXd& u) const {return discrete_partial(p_mask, u_mask, u, 0);} //discrete partial_x (col-vector)
		inline VectorXd By(const VectorXd& v) const {return discrete_partial(p_mask, v_mask, v, 1);} //discrete partial_y (col-vector)
		inline VectorXd Bz(const VectorXd& w) const {return discrete_partial(p_mask, w_mask, w, 2);} //discrete partial_z (col-vector)

		//laplacians
		VectorXd discrete_laplacian(const StructuredPoints& mask, const VectorXd& vals) const;
		inline VectorXd Au(const VectorXd& u) const {return discrete_laplacian(u_mask, u);} //discrete laplacian on u-DOFs
		inline VectorXd Av(const VectorXd& v) const {return discrete_laplacian(v_mask, v);} //discrete laplacian on u-DOFs
		inline VectorXd Aw(const VectorXd& w) const {return discrete_laplacian(w_mask, w);} //discrete laplacian on u-DOFs
		inline VectorXd BB_t(const VectorXd& vals) const {return Bx(Bx_t(vals))+By(By_t(vals))+Bz(Bz_t(vals));}

		
		//update variables by Distributive Gauss-Seidel Relaxation (DGS)
		void GS_discrete_laplacian(const StructuredPoints& mask, VectorXd& x, const VectorXd& b);



		// void DGS();

		//save solution interpolated to pressure DOFs
		void saveas(const std::string filename) const;
	};
}


#endif