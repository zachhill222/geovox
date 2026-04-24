#pragma once

#include "fem/forms/bilinear_H1.hpp"
#include "fem/forms/bilinear_L2.hpp"
#include "fem/forms/linear_L2.hpp"
#include "fem/handlers/dofhandler_charms.hpp" //TODO: replace with a multigrid specialization
#include "fem/handlers/bc_handler.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"
#include "mesh/voxel_mesh.hpp"

#include "util/concepts.hpp"

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
	//
	// We use the trial spaces Q1-iso-Q2 for u,v,w and Q1 for p.
	// This is implemented by refining the geometry to depth d and setting the
	// pressure Q1 dofs. Then for each active element, we refine and activate all of its children at depth d+1.
	// Then the u,v,w dofs are simply the Q1 dofs at this depth d+1.

	template<uint64_t BC=7, uint64_t MAX_DEPTH=10>
	class Stokes
	{
		using Mesh_t  		= VoxelMesh<MAX_DEPTH,false>; //TODO: use Morton order and mesh coloring
		using DofKey_t 		= typename Mesh_t::VoxelVertex::PeriodicVariant<BC>;
		using DOF_t    		= VoxelQ1<DofKey_t>;
		using Handler_t 	= DofHandlerCharms<Mesh_t,DOF_t>; //TODO: replace with multigrid handler?
		using BCHandler_t	= BCHandler<DOF_t>;

		template<typename Action_type>
		using BilinH1_t 	= SymmetricH1<Handler_t,Action_type>;
		template<typename Action_type>
		using BilinL2_t 	= SymmetricL2<Handler_t,Action_type>;
		template<typename Action_type>
		using LinearL2_t	= LinearL2<Handler_t,Action_type>;
		template<typename Action_type>
		using Kernel_t 		= Kernel<4,TypeList<BilinH1_t<Action_type>,BilinL2_t<Action_type>>, TypeList<LinearL2_t<Action_type>>>;

		using Vec_t 		= Eigen::VectorXd;

		Mesh_t 			mesh;
		Handler_t 		velocity_handler, pressure_handler; //all velocity dofs are the same
		BCHandler_t		v_bc, p_bc;
		Vec_t 			u,v,w,p;

		void set_depth(const uint64_t depth) {
			pressure_handler.set_depth(depth);
			velocity_handler.set_depth(depth+1);

			const auto np = pressure_handler.n_dofs();
			const auto nv = velocity_handler.n_dofs();

			p = Vec_t::Zeros(np);
			u = Vec_t::Zeros(nv);
			v = Vec_t::Zeros(nv);
			w = Vec_t::Zeros(nv);
		}

		




	}
}

