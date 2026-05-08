#pragma once

#include "fem/forms/bilinear_H1.hpp"
#include "fem/forms/bilinear_L2.hpp"
#include "fem/forms/bilinear_Hdiv.hpp"
#include "fem/forms/linear_L2.hpp"
#include "fem/forms/form_actions.hpp"

#include "fem/numerics/kernel.hpp"

#include "fem/handlers/dofhandler_charms.hpp" //TODO: replace with a multigrid specialization
#include "fem/handlers/bc_handler.hpp"

#include "fem/dofs/voxel_dof_Q1.hpp"

#include "mesh/voxel_mesh.hpp"

#include "util/concepts.hpp"
#include "util/point.hpp"

#include <span>
#include <vector>
#include <string>
#include <fstream>

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
	// Following the paper "Efficient Uzawa algorithms with projection strategies for geodynamic Stokes flow" by
	// Jang, Lee, Thieulot, Choi, and So, we use the notation
	//
	// [K   G][U]	[F]
	// [G^T 0][P] = [H]
	//
	// Where U is the concatenation of u, v, w and similar for F, and K is the block diagonal matrix with diagonal blocks mu*A_s and
	// G is the (tall) gradient matrix [B_x^T B_y^T B_z^T]
	//
	// We use the trial spaces Q1-iso-Q2 for u,v,w and Q1 for p.
	// This is implemented by refining the geometry to depth d and setting the
	// pressure Q1 dofs. Then for each active element, we refine and activate all of its children at depth d+1.
	// Then the u,v,w dofs are simply the Q1 dofs at this depth d+1.

	template<uint64_t V_BC=7, uint64_t P_BC=0, uint64_t MAX_DEPTH=8>
	class Stokes
	{
		public:
		using Mesh_t  		= VoxelMesh<MAX_DEPTH>; //TODO: use Morton order and mesh coloring
		using Elem_t        = typename Mesh_t::VoxelElement;
		using Vert_t  		= typename Mesh_t::VoxelVertex;
		using V_DofKey_t 	= typename Mesh_t::VoxelVertex::PeriodicVariant<V_BC>;
		using P_DofKey_t 	= typename Mesh_t::VoxelVertex::PeriodicVariant<P_BC>;
		using V_DOF_t    	= VoxelQ1<V_DofKey_t>;
		using P_DOF_t    	= VoxelQ1<P_DofKey_t>;
		using V_Handler_t 	= DofHandlerCharms<Mesh_t,V_DOF_t>; //TODO: replace with multigrid handler?
		using P_Handler_t 	= DofHandlerCharms<Mesh_t,P_DOF_t>; //TODO: replace with multigrid handler?
		using BCHandler_t	= BCHandler<V_DOF_t>;

		template<typename Action_type>
		using BilinL2_t 	= SymmetricL2<P_Handler_t,Action_type>;
		template<typename Action_type>
		using BilinH1_t 	= SymmetricH1<V_Handler_t,Action_type>;
		template<int component, typename Action_type>
		using BilinHdiv_t   = BilinearHdiv<V_Handler_t,P_Handler_t,component,Action_type>; //for b(V,q) = -int(div(V)*q) with vector test functions V
		template<int component, typename Action_type>
		using BilinHdivAdj_t = BilinearHdivAdjoint<P_Handler_t,V_Handler_t,component,Action_type>; //for b(U,q) = -int(div(U)*q) with scalar test function

		Mesh_t 			mesh;
		V_Handler_t 	velocity_handler; //all velocity dofs are the same
		P_Handler_t		pressure_handler; 

		std::vector<double> U, P;
		BCHandler_t		u_bc, v_bc, w_bc, p_bc;
		double 			mu     = 1.0; //viscosity

		Stokes() : mesh{{0,0,0},{1,1,1}}, velocity_handler{mesh}, pressure_handler{mesh} {}

		void set_depth(const uint64_t depth) {
			mesh.set_depth(depth);
			pressure_handler.set_depth(depth);

			mesh.set_depth(depth+1);
			velocity_handler.set_depth(depth+1);

			const auto np = pressure_handler.n_dofs();
			const auto nv = velocity_handler.n_dofs();

			P.assign(np,0.0);
			U.assign(3*nv,0.0);
		}

		//apply dirichlet BC for velocity
		template<typename Predicate, typename Function = std::nullptr_t>
		void add_velocity_bc(Predicate&& pred, Function&& fun = nullptr) {
			if constexpr (NULLPTR_T<Function>) {
				u_bc.add_essential(pred, [](V_DOF_t dof) {return 0.0;});
				v_bc.add_essential(pred, [](V_DOF_t dof) {return 0.0;});
				w_bc.add_essential(pred, [](V_DOF_t dof) {return 0.0;});
			}
			else {
				u_bc.add_essential(pred, [fun](V_DOF_t dof) {return fun(dof)[0];});
				v_bc.add_essential(pred, [fun](V_DOF_t dof) {return fun(dof)[1];});
				w_bc.add_essential(pred, [fun](V_DOF_t dof) {return fun(dof)[2];});
			}
		}

		//cache the dofs that need the dirichlet bc
		void cache_bc() {
			u_bc.cache(velocity_handler.curr_compressed_dofs());
			v_bc.cache(velocity_handler.curr_compressed_dofs());
			w_bc.cache(velocity_handler.curr_compressed_dofs());
		}

		//compute K*U (all three velocities times their corresponding stiffness matrix)
		void K_U(std::span<double> KU, std::span<const double> U) const {
			assert(KU.size()>0);
			assert(KU.size() == U.size());
			assert(KU.size() == 3*velocity_handler.n_dofs());


			//get indices for subspans for u, v, w components
			assert(U.size()%3 == 0);
			const auto N = U.size() / 3;

			//only need one bilinear form
			using Kernel_type = Kernel<4, TypeList<BilinH1_t<MatVecAction>>>;
			BilinH1_t<MatVecAction> A_form(velocity_handler);
			Kernel_type kernel(A_form);

			//set up integrating action over each element
			//TODO: is having three kernels in parallel better?
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				A_form.set_basis(v_dofs, v_dofs);
				kernel.compute_all();	//compute local stiffness matrix
				
				//set each component, multiply, scatter
				A_form.set_vecs(KU.subspan(0,N), U.subspan(0,N));
				A_form.multiply();
				A_form.scatter();

				A_form.set_vecs(KU.subspan(N,N), U.subspan(N,N));
				A_form.multiply();
				A_form.scatter();

				A_form.set_vecs(KU.subspan(2*N,N), U.subspan(2*N,N));
				A_form.multiply();
				A_form.scatter();
			};

			//set up predicate to only integrate over active elements (natural BC)
			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};

			//perform the loop
			//TODO: parallel by element color?
			mesh.template for_each<Elem_t>(action, false, predicate);

			//scale result by the viscosity
			for (size_t i=0; i<KU.size(); ++i) {
				KU[i] *= mu;
			}

			//apply the boundary conditions to treat each corresponding row as an identity row
			u_bc.apply_matvec(KU.subspan(0,N),   U.subspan(0,N));
			v_bc.apply_matvec(KU.subspan(N,N),   U.subspan(N,N));
			w_bc.apply_matvec(KU.subspan(2*N,N), U.subspan(2*N,N));
		}

		//compute some number of iterations of Gauss-Seidel (forwards or backwards) on K*U=F
		//using the fact that K is block-diagonal with the same matrix for each block
		template<bool FORWARD=true>
		void K_inv_gs(std::span<double> U, std::span<const double> F, const int n_steps=1) const {
			assert(U.size()>0);
			assert(U.size() == F.size());
			assert(U.size() == 3*velocity_handler.n_dofs());


			//get indices for subspans for u, v, w components
			assert(U.size()%3 == 0);
			const auto N = U.size() / 3;

			//only need one bilinear form
			using Kernel_type = Kernel<4, TypeList<BilinH1_t<MatVecAction>>>;
			BilinH1_t<MatVecAction> A_form(velocity_handler);
			Kernel_type kernel(A_form);

			//set up integrating action over each element
			//TODO: is having three kernels in parallel better?
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				A_form.set_basis(v_dofs, v_dofs);
				kernel.compute_all();	//compute local stiffness matrix
				
				//set each component, multiply, scatter
				A_form.set_vecs(U.subspan(0,N), F.subspan(0,N));
				A_form.template gauss_seidel<FORWARD>();
				A_form.scatter();

				A_form.set_vecs(U.subspan(N,N), F.subspan(N,N));
				A_form.template gauss_seidel<FORWARD>();
				A_form.scatter();

				A_form.set_vecs(U.subspan(2*N,N), F.subspan(2*N,N));
				A_form.template gauss_seidel<FORWARD>();
				A_form.scatter();
			};

			//set up predicate to only integrate over active elements (natural BC)
			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};

			//perform the loop
			//TODO: parallel by element color?
			for (int n=0; n<n_steps; ++n) {
				mesh.template for_each<Elem_t>(action, false, predicate);
			}

			//scale result by the viscosity
			const double mu_inv = 1.0/mu;
			for (size_t i=0; i<U.size(); ++i) {
				U[i] *= mu_inv;
			}

			//apply the boundary conditions
			apply_velocity_bc(U);
		}

		//compute G*p (result is a vector with the size of U)
		void G_P(std::span<double> GP, std::span<const double> P) const {
			assert(GP.size()>0);
			assert(GP.size() == 3*velocity_handler.n_dofs());
			assert(P.size()  ==   pressure_handler.n_dofs());

			//get indices for subspans for u, v, w components
			assert(GP.size()%3 == 0);
			const auto N = GP.size() / 3;

			//set up the bilinear form for each velocity (test dof) component
			using Kernel_type = Kernel<4, TypeList<BilinHdiv_t<0,MatVecAction>, BilinHdiv_t<1,MatVecAction>, BilinHdiv_t<2,MatVecAction>>>;
			BilinHdiv_t<0,MatVecAction> Bx_form(velocity_handler, pressure_handler);
			BilinHdiv_t<1,MatVecAction> By_form(velocity_handler, pressure_handler);
			BilinHdiv_t<2,MatVecAction> Bz_form(velocity_handler, pressure_handler);
			Kernel_type kernel(Bx_form, By_form, Bz_form);

			//assign subspans to each form
			Bx_form.set_vecs(GP.subspan(0,N),   P);
			By_form.set_vecs(GP.subspan(N,N),   P);
			Bz_form.set_vecs(GP.subspan(2*N,N), P);

			//define integration action
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				const auto p_dofs = pressure_handler.basis_active(el);

				Bx_form.set_basis(v_dofs, p_dofs);
				By_form.set_basis(v_dofs, p_dofs);
				Bz_form.set_basis(v_dofs, p_dofs);
				
				kernel.compute_all();
				
				//set each component, multiply, scatter
				Bx_form.multiply();
				By_form.multiply();
				Bz_form.multiply();
				
				kernel.scatter_all();
			};

			//set up predicate to only integrate over active elements (natural BC)
			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};

			//perform the loop
			//TODO: parallel by element color?
			mesh.template for_each<Elem_t>(action, false, predicate);
		}


		//compute GT*U (result is a vector with the size of P)
		void GT_U(std::span<double> GTU, std::span<const double> U) const {
			assert(GTU.size()>0);
			assert(GTU.size() ==   pressure_handler.n_dofs());
			assert(U.size()   == 3*velocity_handler.n_dofs());

			//get indices for subspans for u, v, w components
			assert(U.size()%3 == 0);
			const auto N = U.size() / 3;

			//set up the bilinear form for each velocity (trial dof) component
			using Kernel_type = Kernel<4, TypeList<BilinHdivAdj_t<0,MatVecAction>, BilinHdivAdj_t<1,MatVecAction>, BilinHdivAdj_t<2,MatVecAction>>>;
			BilinHdivAdj_t<0,MatVecAction> Bx_t_form(pressure_handler, velocity_handler);
			BilinHdivAdj_t<1,MatVecAction> By_t_form(pressure_handler, velocity_handler);
			BilinHdivAdj_t<2,MatVecAction> Bz_t_form(pressure_handler, velocity_handler);
			Kernel_type kernel(Bx_t_form, By_t_form, Bz_t_form);

			//assign subspans to each form
			Bx_t_form.set_vecs(GTU, U.subspan(0,N));
			By_t_form.set_vecs(GTU, U.subspan(N,N));
			Bz_t_form.set_vecs(GTU, U.subspan(2*N,N));

			//define integration action
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				const auto p_dofs = pressure_handler.basis_active(el);

				Bx_t_form.set_basis(p_dofs, v_dofs);
				By_t_form.set_basis(p_dofs, v_dofs);
				Bz_t_form.set_basis(p_dofs, v_dofs);
				
				kernel.compute_all();
				
				//set each component, multiply, scatter
				Bx_t_form.multiply();
				By_t_form.multiply();
				Bz_t_form.multiply();

				kernel.scatter_all();
			};

			//set up predicate to only integrate over active elements (natural BC)
			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};

			//perform the loop
			//TODO: parallel by element color?
			mesh.template for_each<Elem_t>(action, false, predicate);
		}

		//compute M*P (pressure component of the block diagonal preconditioner)
		void M_P(std::span<double> MP, std::span<const double> P) const {
			assert(MP.size()>0);
			assert(MP.size() == P.size());
			assert(MP.size() == pressure_handler.n_dofs());

			using Kernel_type = Kernel<4, TypeList<BilinL2_t<MatVecAction>>>;
			BilinL2_t<MatVecAction> M_form(velocity_handler);
			Kernel_type kernel(M_form);

			//set up integrating action over each element
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				M_form.set_basis(v_dofs, v_dofs);
				kernel.compute_all();	//compute local stiffness matrix
				
				//set each component, multiply, scatter
				M_form.set_vecs(MP,P);
				M_form.multiply();
				M_form.scatter();
			};

			//set up predicate to only integrate over active elements (natural BC)
			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};

			//perform the loop
			//TODO: parallel by element color?
			mesh.template for_each<Elem_t>(action, false, predicate);
		}

		//compute some number of iterations of Gauss-Seidel (forwards or backwards) on M*P=H
		//using the fact that K is block-diagonal with the same matrix for each block
		template<bool FORWARD=true>
		void M_inv_gs(std::span<double> P, std::span<const double> H, const int n_steps=1) const {
			assert(P.size()>0);
			assert(P.size() == H.size());
			assert(P.size() == pressure_handler.n_dofs());

			//get indices for subspans for u, v, w components
			using Kernel_type = Kernel<4, TypeList<BilinL2_t<MatVecAction>>>;
			BilinL2_t<MatVecAction> M_form(velocity_handler);
			Kernel_type kernel(M_form);

			//set up integrating action over each element
			//TODO: is having three kernels in parallel better?
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				M_form.set_basis(v_dofs, v_dofs);
				kernel.compute_all();	//compute local stiffness matrix
				
				//set each component, multiply, scatter
				M_form.set_vecs(P, H);
				M_form.template gauss_seidel<FORWARD>();
				M_form.scatter();
			};

			//set up predicate to only integrate over active elements (natural BC)
			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};

			//perform the loop
			//TODO: parallel by element color?
			for (int n=0; n<n_steps; ++n) {
				mesh.template for_each<Elem_t>(action, false, predicate);
			}
		}

		//apply the velocity BC
		void apply_velocity_bc(std::span<double> U) const {
			assert(U.size() == 3*velocity_handler.n_dofs());
			const size_t N = U.size()/3;
			u_bc.apply(U.subspan(0,  N), as_span(velocity_handler.curr_compressed_dofs()));
			v_bc.apply(U.subspan(N,  N), as_span(velocity_handler.curr_compressed_dofs()));
			w_bc.apply(U.subspan(2*N,N), as_span(velocity_handler.curr_compressed_dofs()));
		}

		//apply a standard Uzawa iterations using some number of inner Gauss-Seidel iterations
		template<bool FORWARD=true>
		void standard_uzawa(std::span<double> U, std::span<double> P, std::span<const double> F, std::span<const double> H, const double w, const int n) const {
			LogTime timer{"Stokes::standard_uzawa"};
			assert(U.size() == F.size());
			assert(P.size() == H.size());
			assert(w>0.0);

			//update U: K*U = F - G*P
			std::vector<double> rhs(U.size(), 0.0);
			G_P(rhs, P);
			#pragma omp simd
			for (size_t i=0; i<rhs.size(); ++i) {
				rhs[i] = F[i] - rhs[i];
			}
			K_inv_gs<FORWARD>(U, rhs, n);

			//update P: P += w*(G^T * U - H)
			rhs.assign(P.size(), 0.0);
			GT_U(rhs, U);
			P[0] = 0.0;
			#pragma omp simd
			for (size_t i=1; i<rhs.size(); ++i) {
				P[i] += w*(rhs[i] - H[i]);
			}
		}

		//save solution
		void save_as(const std::string filename) const {
			LogTime timer{"Stokes::save_as"};

			std::ofstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error("Stokes::save_as - could not open file: " + filename);
			}

			//write the mesh and get the number of vertices
			const auto n_verts = mesh.write_unstructured_vtk(file);

			//interpolate the solution to the vertex values
			auto p_vals = pressure_handler.interpolate_to_vertices(P, n_verts);

			const auto N = U.size()/3;
			auto u_vals = velocity_handler.interpolate_to_vertices(as_span(U, 0,  N), n_verts);
			auto v_vals = velocity_handler.interpolate_to_vertices(as_span(U, N,  N), n_verts);
			auto w_vals = velocity_handler.interpolate_to_vertices(as_span(U, 2*N,N), n_verts);

			//append solution header
			file << "POINT_DATA " << n_verts << "\n"
				 << "FIELD solution 2\n";

			mesh.append_unstructured_point_data_vtk(
				file,
				"pressure 1 " + std::to_string(n_verts) + " float",
				n_verts,
				[&](Vert_t vtx) {return p_vals[vtx.linear_index()];});

			mesh.append_unstructured_point_data_vtk(
				file,
				"velocity 3 " + std::to_string(n_verts) + " float",
				n_verts,
				[&](Vert_t vtx) {return Point<3,float>{
							u_vals[vtx.linear_index()], 
							v_vals[vtx.linear_index()], 
							w_vals[vtx.linear_index()]};
						}
			);
		}
	};


	



	
}

