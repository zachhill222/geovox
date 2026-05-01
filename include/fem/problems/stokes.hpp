#pragma once

#include "fem/forms/bilinear_H1.hpp"
#include "fem/forms/bilinear_L2.hpp"
#include "fem/forms/linear_L2.hpp"
#include "fem/forms/form_actions.hpp"

#include "fem/handlers/dofhandler_charms.hpp" //TODO: replace with a multigrid specialization
#include "fem/handlers/bc_handler.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"
#include "mesh/voxel_mesh.hpp"

#include "util/concepts.hpp"

#include <span>
#include <vector>

#include <Eigen/SparseCore>


namespace GV
{
	// A class to solve the stokes problem
	//
	//	-mu*laplace(U) + grad(p) = f
	//		div(U)				 = 0
	// 
	// where U = (u,v,w) is the flow velocity and
	// p is the flow pressure.
	//
	// Multiplying by a test function psi in H1^3 and integrating gives us a weak form
	//
	// mu*int_D (grad U : grad psi) - int_D (p*div(psi)) = int_D f * psi (i=0,1,2)
	// int_D(div(U)*psi) = 0
	//
	// where grad U : grad psi = sum( U[i]_j * psi[i]_j)
	// where X[i]_j is the partial derivative of the i-th component of X along the jth axis.
	// examining component by component, we may write our system in block form as
	//
	//	[mu*A_s	0		0		B_x^T][u]	[f_x]
	//	[0		mu*A_s	0		B_y^T][v]	[f_y]
	//	[0		0		mu*A_s	B_z^T][w] =	[f_z]
	//	[B_x	B_y		B_z		0	 ][p]	[0  ]
	//
	//
	// Where each A_s is the standard symmetric H1-H1 stiffness matrix and
	// each B_* is a non-symmetric L2-L2 mass matrix with entries
	// B_*_ij = int_D( phi_j * psi_i ) where phi_j is a pressure trial function
	// and psi_i is a velocity test function.
	// 
	// Note that because the dofs for u,v,w are the same, B_x, B_y, and B_z are identical.
	//
	// We use the trial spaces Q1-iso-Q2 for u,v,w and Q1 for p.
	// This is implemented by refining the geometry to depth d and setting the
	// pressure Q1 dofs. Then for each active element, we refine and activate all of its children at depth d+1.
	// Then the u,v,w dofs are simply the Q1 dofs at this depth d+1.

	template<uint64_t BC=7, uint64_t MAX_DEPTH=10>
	class Stokes
	{
		using Mesh_t  		= VoxelMesh<MAX_DEPTH,false>; //TODO: use Morton order and mesh coloring
		using Elem_t        = typename Mesh_t::VoxelElement;
		using DofKey_t 		= typename Mesh_t::VoxelVertex::PeriodicVariant<BC>;
		using DOF_t    		= VoxelQ1<DofKey_t>;
		using Handler_t 	= DofHandlerCharms<Mesh_t,DOF_t>; //TODO: replace with multigrid handler?
		using BCHandler_t	= BCHandler<DOF_t>;

		template<typename Action_type>
		using BilinH1_t 	= SymmetricH1<Handler_t,Action_type>;
		template<typename Action_type>
		using BilinL2_t 	= BilinearL2<Handler_t,false,Action_type>;
		template<typename Action_type>
		using LinearL2_t	= LinearL2<Handler_t,Action_type>;
		template<typename Action_type>
		using Kernel_t 		= Kernel<4,TypeList<BilinH1_t<Action_type>,BilinL2_t<Action_type>>, TypeList<LinearL2_t<Action_type>>>;

		using Vec_t 		= Eigen::VectorXd;

		Mesh_t 			mesh;
		Handler_t 		velocity_handler, pressure_handler; //all velocity dofs are the same
		BCHandler_t		v_bc, p_bc;
		// Vec_t 			u,v,w,p;
		double 			mu     = 0.001; //viscisity

		void set_depth(const uint64_t depth) {
			pressure_handler.set_depth(depth);
			velocity_handler.set_depth(depth+1);

			const auto np = pressure_handler.n_dofs();
			const auto nv = velocity_handler.n_dofs();

			// p = Vec_t::Zeros(np);
			// u = Vec_t::Zeros(nv);
			// v = Vec_t::Zeros(nv);
			// w = Vec_t::Zeros(nv);
		}

		//apply n steps of smoothing using forward iterations for velocity
		template<bool FORWARD>
		void smooth(
				int n_steps,
				std::span<double> u,
				std::span<double> v,
				std::span<double> w,
				std::span<const double> res_x,
				std::span<const double> res_y,
				std::span<const double> res_z,
				std::span<const double> res_p) const {
			
			//sanity check inputs
			assert(res_x.size() == res_y.size());
			assert(res_x.size() == res_z.size());
			assert(res_x.size() == velocity_handler.n_dofs());
			assert(res_p.size() == pressure_handler.n_dofs());
			assert(res_x.size() == u.size());
			assert(res_y.size() == v.size());
			assert(res_z.size() == w.size());

			//build the forms and kernel to smooth the velocity dofs
			using Kernel_type = Kernel<4,
					TypeList<
						BilinH1_t<MatVecAction>,  //Au
						BilinH1_t<MatVecAction>,  //Av
						BilinH1_t<MatVecAction>,  //Aw
						BilinL2_t<MatVecAction>>; //Bp_transpose

			//todo: loop in parallel per-color
			BilinH1_t<MatVecAction> Au(velocity_handler), Av(velocity_handler), Aw(velocity_handler);
			BilinL2_t<MatVecAction> Bp(velocity_handler, pressure_handler);

			const auto diag = mesh.high - mesh.low;
			Kernel_type kernel(diag[0], diag[1], diag[2], Au, Av, Aw, Bp);

			//assign storage and assemble the rhs
			std::vector<double> bp(res_x.size(), 0.0);

			assert(p.size() == res_p.n_dofs());
			auto bp_action = [&, this](Elem_t el) {
				const auto p_dofs = pressure_handler.basis_active(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				kernel.set_element(el);
				Bp.set_basis(p_dofs, v_dofs);
				kernel.template B_compute<3>(); //set up local matrix
				Bp.multiply(); 					//compute local contribution to B*p
				Bp.scatter();					//accumulate local contribution into the global vector
			};

			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};
			
			Bp.set_vecs(bp,p); //current pressure
			mesh.for_each<Elem_t>(bp_action, predicate);

			//assemble the rhs vectors
			std::vector<double> rhs_x(res_x.size());
			std::vector<double> rhs_y(res_y.size());
			std::vector<double> rhs_z(res_z.size());
			
			const double mu_1 = 1.0/mu;
			for (size_t i=0; i<rhs_x.size(); ++i) {
				rhs_x[i] = mu_1 * (res_x[i] - bp[i]);
				rhs_y[i] = mu_1 * (res_y[i] - bp[i]);
				rhs_z[i] = mu_1 * (res_z[i] - bp[i]);
			}

			//smooth each of the velocity dofs
			auto action = [&, this](Elem_t el) {
				const auto v_dofs = velocity_handler.basis_active(el);
				kernel.set_element(el);
				Au.set_basis(v_dofs, v_dofs);
				Av.set_basis(v_dofs, v_dofs);
				Aw.set_basis(v_dofs, v_dofs);
				kernel.template B_compute<0>(); //Au
				kernel.template B_compute<1>(); //Av
				kernel.template B_compute<2>(); //Aw

				if constexpr (FORWARD) {
					Au.gauss_seidel();
					Av.gauss_seidel();
					Aw.gauss_seidel();
				}
				else {
					Au.gauss_seidel_backwards();
					Av.gauss_seidel_backwards();
					Aw.gauss_seidel_backwards();
				}

				Au.scatter();
				Av.scatter();
				Aw.scatter();
			};

			Au.set_vecs(u,rhs_x);
			Av.set_vecs(v,rhs_y);
			Aw.set_vecs(w,rhs_z);
			for (int n=0; n<n_steps; ++n) {
				//todo change to parallel over colors
				mesh.for_each<Elem_t>(action, predicate);
			}

			//smooth the pressure
		}

		//solve the system on a coarse grid
		void smooth(
				std::span<double> u,
				std::span<double> v,
				std::span<double> w,
				std::span<double> p,
				std::span<const double> res_x,
				std::span<const double> res_y,
				std::span<const double> res_z,
				std::span<const double> res_p) const {
			
			//sanity check inputs
			assert(res_x.size() == res_y.size());
			assert(res_x.size() == res_z.size());
			assert(res_x.size() == velocity_handler.n_dofs());
			assert(res_p.size() == pressure_handler.n_dofs());
			assert(res_x.size() == u.size());
			assert(res_y.size() == v.size());
			assert(res_z.size() == w.size());

			//build the forms and kernel to smooth the velocity dofs
			using Kernel_type = Kernel<4,
					TypeList<
						BilinH1_t<MatVecAction>,  //Au
						BilinH1_t<MatVecAction>,  //Av
						BilinH1_t<MatVecAction>,  //Aw
						BilinL2_t<MatVecAction>>; //Bp_transpose

			//todo: loop in parallel per-color
			BilinH1_t<MatVecAction> Au(velocity_handler), Av(velocity_handler), Aw(velocity_handler);
			BilinL2_t<MatVecAction> Bp(velocity_handler, pressure_handler);

			const auto diag = mesh.high - mesh.low;
			Kernel_type kernel(diag[0], diag[1], diag[2], Au, Av, Aw, Bp);

			//assign storage and assemble the rhs
			std::vector<double> bp(res_x.size(), 0.0);

			assert(p.size() == res_p.n_dofs());
			auto bp_action = [&, this](Elem_t el) {
				const auto p_dofs = pressure_handler.basis_active(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				kernel.set_element(el);
				Bp.set_basis(p_dofs, v_dofs);
				kernel.template B_compute<3>(); //set up local matrix
				Bp.multiply(); 					//compute local contribution to B*p
				Bp.scatter();					//accumulate local contribution into the global vector
			};

			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};
			
			Bp.set_vecs(bp,p); //current pressure
			mesh.for_each<Elem_t>(bp_action, predicate);

			//assemble the rhs vectors
			std::vector<double> rhs_x(res_x.size());
			std::vector<double> rhs_y(res_y.size());
			std::vector<double> rhs_z(res_z.size());
			
			const double mu_1 = 1.0/mu;
			for (size_t i=0; i<rhs_x.size(); ++i) {
				rhs_x[i] = mu_1 * (res_x[i] - bp[i]);
				rhs_y[i] = mu_1 * (res_y[i] - bp[i]);
				rhs_z[i] = mu_1 * (res_z[i] - bp[i]);
			}

			//smooth each of the velocity dofs
			auto action = [&, this](Elem_t el) {
				const auto v_dofs = velocity_handler.basis_active(el);
				kernel.set_element(el);
				Au.set_basis(v_dofs, v_dofs);
				Av.set_basis(v_dofs, v_dofs);
				Aw.set_basis(v_dofs, v_dofs);
				kernel.template B_compute<0>(); //Au
				kernel.template B_compute<1>(); //Av
				kernel.template B_compute<2>(); //Aw

				if constexpr (FORWARD) {
					Au.gauss_seidel();
					Av.gauss_seidel();
					Aw.gauss_seidel();
				}
				else {
					Au.gauss_seidel_backwards();
					Av.gauss_seidel_backwards();
					Aw.gauss_seidel_backwards();
				}

				Au.scatter();
				Av.scatter();
				Aw.scatter();
			};

			Au.set_vecs(u,rhs_x);
			Av.set_vecs(v,rhs_y);
			Aw.set_vecs(w,rhs_z);
			for (int n=0; n<n_steps; ++n) {
				//todo change to parallel over colors
				mesh.for_each<Elem_t>(action, predicate);
			}
		}
	}
}

