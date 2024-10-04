#ifndef MAC_H
#define MAC_H

#define MAC_DOMAIN_MARKER 0

#include "mesh/vtk_structured.hpp"
#include "geometry/assembly.hpp"
#include "util/point.hpp"
#include "util/box.hpp"
#include "solvers/conjugate_gradient.hpp"

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <functional> //for passing method functions to solvers

using StructuredPoints = GeoVox::mesh::StructuredPoints;
using Assembly = GeoVox::geometry::Assembly;

namespace GeoVox::mac{
	class MacMesh{
	public:
		MacMesh() {}
		
		MacMesh(const Box& box, const long unsigned int N[3], const Assembly &assembly){
			//get spacing between DOFs
			Point3 H = (box.high()-box.low())/Point3(N[0], N[1], N[2]);
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
			p = std::vector<double>( N[0]    *  N[1]    *  N[2]    , 0);
			u = std::vector<double>((N[0]+1) *  N[1]    *  N[2]    , 0);
			v = std::vector<double>( N[0]    * (N[1]+1) *  N[2]    , 0);
			w = std::vector<double>( N[0]    *  N[1]    * (N[2]+1) , 0);

			//initialize forcing terms to 0
			f1 = std::vector<double>(u.size(), 0);
			f2 = std::vector<double>(v.size(), 0);
			f3 = std::vector<double>(w.size(), 0);
			g  = std::vector<double>(p.size(), 0);
		}

		MacMesh(const Box& box, const long unsigned int N[3]){
			//get spacing between DOFs
			Point3 H = (box.high()-box.low())/Point3(N[0], N[1], N[2]);
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
			p = std::vector<double>( N[0]    *  N[1]    *  N[2]   );
			u = std::vector<double>((N[0]+1) *  N[1]    *  N[2]   );
			v = std::vector<double>( N[0]    * (N[1]+1) *  N[2]   );
			w = std::vector<double>( N[0]    *  N[1]    * (N[2]+1));
		}

		//DOF masks
		StructuredPoints p_mask; //pressure mask (global indexing): Nx by Ny by Nz
		StructuredPoints u_mask; //x-velocity mask (global indexing): (Nx+1) by Ny by Nz
		StructuredPoints v_mask; //y-velocity mask (global indexing): Nx by (Ny+1) by Nz
		StructuredPoints w_mask; //z-velocity mask (global indexing): Nx by Ny by (Nz+1)

		//DOF values
		std::vector<double> p; //pressure values (reduced indexing): number of nonzeros in p_mask
		std::vector<double> u; //x-velocity values (reduced indexing): number of nonzeros in
		std::vector<double> v; //y-velocity values (reduced indexing)
		std::vector<double> w; //z-velocity values (reduced indexing)

		//RHS values
		std::vector<double> f1; //forcing term in x-direction at x-DOFs
		std::vector<double> f2; //forcing term in y-direction at y-DOFs
		std::vector<double> f3; //forcing term in z-direction at z-DOFsS
		std::vector<double> g;  //forcing term for p at p-DOFs
		

		//laplacians
		std::vector<double> discrete_laplacian(const StructuredPoints &mask, const std::vector<double> &vals) const;
		inline std::vector<double> Au(const std::vector<double> &u) const {return discrete_laplacian(u_mask, u);} //discrete laplacian on u-DOFs
		inline std::vector<double> Av(const std::vector<double> &v) const {return discrete_laplacian(v_mask, v);} //discrete laplacian on u-DOFs
		inline std::vector<double> Aw(const std::vector<double> &w) const {return discrete_laplacian(w_mask, w);} //discrete laplacian on u-DOFs

		//gradients of pressure
		std::vector<double> discrete_partial_t(const StructuredPoints &eval_dof_mask, const StructuredPoints &given_dof_mask, const std::vector<double> &vals, const int direction) const;
		inline std::vector<double> Bx_t(const std::vector<double> &p) const {return discrete_partial(u_mask, p_mask, p, 0);} //discrete partial_x (row-vector)
		inline std::vector<double> By_t(const std::vector<double> &p) const {return discrete_partial(v_mask, p_mask, p, 1);} //discrete partial_y (row-vector)
		inline std::vector<double> Bz_t(const std::vector<double> &p) const {return discrete_partial(w_mask, p_mask, p, 2);} //discrete partial_z (row-vector)

		//divergence of velocity
		std::vector<double> discrete_partial(const StructuredPoints &eval_dof_mask, const StructuredPoints &given_dof_mask, const std::vector<double> &vals, const int direction) const;
		inline std::vector<double> Bx(const std::vector<double> &u) const {return discrete_partial(p_mask, u_mask, u, 0);} //discrete partial_x (col-vector)
		inline std::vector<double> By(const std::vector<double> &v) const {return discrete_partial(p_mask, v_mask, v, 1);} //discrete partial_y (col-vector)
		inline std::vector<double> Bz(const std::vector<double> &w) const {return discrete_partial(p_mask, w_mask, w, 2);} //discrete partial_z (col-vector)

		//update variables by Distributive Gauss-Seidel Relaxation (DGS)
		std::vector<double> BB_t(const std::vector<double> &vals) const;
		void DGS();

		//save solution interpolated to pressure DOFs
		void saveas(const std::string filename) const;
	};
}


#endif