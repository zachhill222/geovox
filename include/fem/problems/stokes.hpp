#pragma once

#include "fem/problems/stokes_wrapper_eigen.hpp" //use the methods in this class for matrix-free iterative solvers in Eigen

#include "fem/forms/bilinear/matrix_multiply.hpp"
#include "fem/forms/bilinear/matrix_jacobi.hpp"
#include "fem/forms/bilinear/policy_evaluation.hpp"
#include "fem/forms/linear/vector_assembler.hpp"
#include "fem/forms/linear/policy_evaluation.hpp"

#include "fem/numerics/kernel.hpp"

#include "fem/handlers/dofhandler_charms.hpp" //TODO: replace with a multigrid specialization?
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
		using V_BC_Handler_t= BCHandler<V_DOF_t>;
		using P_BC_Handler_t= BCHandler<P_DOF_t>;

		using EvalL2        = BilinearL2<P_DOF_t,P_DOF_t>;
		using BilinL2_t 	= BilinearFormMultiply<P_Handler_t,P_Handler_t,EvalL2>;
		using BilinL2Jacobi = BilinearFormJacobi<P_Handler_t,P_Handler_t,EvalL2>;

		using EvalH1        = BilinearH1<V_DOF_t, V_DOF_t>;
		using BilinH1_t 	= BilinearFormMultiply<V_Handler_t,V_Handler_t,EvalH1>;
		using BilinH1Jacobi = BilinearFormJacobi<V_Handler_t,V_Handler_t,EvalH1>;

		template<int k>
		using EvalHdiv      = BilinearHdiv<V_DOF_t,P_DOF_t,k>;

		template<int k>
		using BilinHdiv_t   = BilinearFormMultiply<V_Handler_t,P_Handler_t,EvalHdiv<k>>; //for b(V,q) = -int(div(V)*q) with vector test functions V
		
		template<int k>
		using EvalHdivAdj   = BilinearHdivAdjoint<P_DOF_t, V_DOF_t,k>;

		template<int k>
		using BilinHdivAdj_t= BilinearFormMultiply<P_Handler_t,V_Handler_t,EvalHdivAdj<k>>; //for b(U,q) = -int(div(U)*q) with scalar test function

		//body force terms
		using LinL2         = LinearFormAssembler<V_Handler_t, LinearL2<V_DOF_t>>;
		double fu{0}, fv{0}, fw{0};

		//define primary problem components
		Mesh_t 			mesh;
		V_Handler_t 	velocity_handler; //all velocity dofs are the same
		P_Handler_t		pressure_handler; 

		//set solution storage
		std::vector<double> X; //U and P combined

		//set boundary condition handlers
		V_BC_Handler_t		u_bc, v_bc, w_bc;
		P_BC_Handler_t		p_bc;

		//define problem parameters. TODO: incoroporate viscosity into the H1 form
		double 				mu = 1.0; //viscosity

		Stokes(double L, double W, double H) :
			mesh{{0,0,0},{L,W,H}}, 
			velocity_handler{mesh}, pressure_handler{mesh} {}

		Stokes() : Stokes(1.0,1.0,1.0) {}
			

		void set_body_force(const double x, const double y, const double z) {
			fu = x;
			fv = y;
			fw = z;
		}

		size_t n_vel_total() const {return 3*velocity_handler.n_dofs();}
		size_t n_vel_individual() const {return velocity_handler.n_dofs();}
		size_t n_pres() const {return pressure_handler.n_dofs();}

		//access individual components of the solution
		std::span<double> u() {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,0,N1);
		}

		std::span<const double> u() const {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,0,N1);
		}

		std::span<double> v() {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,N1,N1);
		}

		std::span<const double> v() const {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,N1,N1);
		}

		std::span<double> w() {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,2*N1,N1);
		}

		std::span<const double> w() const {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,2*N1,N1);
		}

		std::span<double> U() {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,0,3*N1);
		}

		inline auto P() {return P();}

		std::span<double> p() {
			const auto N1 = velocity_handler.n_dofs();
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			return as_span(X,3*N1,N2);
		}

		std::span<const double> p() const {
			const auto N1 = velocity_handler.n_dofs();
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			return as_span(X,3*N1,N2);
		}

		void set_depth(const uint64_t depth) {
			mesh.set_depth(depth);
			pressure_handler.set_depth(depth);
			pressure_handler.compress_dof_numbers();

			mesh.set_depth(depth+1);
			velocity_handler.set_depth(depth+1);
			velocity_handler.compress_dof_numbers();

			const auto np = pressure_handler.n_dofs();
			const auto nv = velocity_handler.n_dofs();
			X.assign(3*nv+np,0.0);
		}

		//TODO: add dof and mesh predicates (i.e., refine low accuracy dofs and only activate relevant elements)
		void refine() {
			LogTime timer{"Stokes::refine"};

			const auto nv_old = velocity_handler.n_dofs();
			const auto np_old = pressure_handler.n_dofs();

			//refine all pressure dofs and update the coefficients
			pressure_handler.refine(pressure_handler.curr_compressed_dofs());
			mesh.process_request_active(); //somewhat unnecessary
			pressure_handler.compress_dof_numbers();

			//for each active pressure dof, activate each child as a velocity dof
			//for each active support element in the pressure, activate all 8 children
			//the element activation is done by request when activating the velocity dofs
			//note that the pressure and velocity handlers are compatible in the sense that keys
			//for pressure dofs are also keys for velocity dofs
			velocity_handler.set_all_inactive();

			for (const P_DOF_t p : pressure_handler.curr_compressed_dofs()) {
				const V_DOF_t v(p.key);
				for (const V_DOF_t child : v.children()) {
					if (child.exists()) {
						velocity_handler.activate(child);
					}
				}
			}
			mesh.process_request_active();
			velocity_handler.compress_dof_numbers();


			//interpolate solution
			std::vector<double> X_old(std::move(X));
			const auto n_v = velocity_handler.n_dofs();
			const auto n_p = pressure_handler.n_dofs();
			X.assign(3*n_v+n_p, 0.0);

			assert(3*nv_old+np_old == X_old.size());
			velocity_handler.update_coefs(u(), as_span(X_old,0,nv_old));
			velocity_handler.update_coefs(v(), as_span(X_old,nv_old,nv_old));
			velocity_handler.update_coefs(w(), as_span(X_old,2*nv_old,nv_old));
			pressure_handler.update_coefs(p(), as_span(X_old,3*nv_old, np_old));
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

		//compute the body force (effectively M*F)
		template<typename Container_t>
		inline void compute_F(Container_t& F) const {compute_F(as_span(F));}

		void compute_F(std::span<double> F) const {
			const auto N = n_vel_individual();
			assert(F.size()==n_vel_total());
			assert(F.size()==3*N);

			LinL2 F_u{velocity_handler, fu};
			LinL2 F_v{velocity_handler, fv};
			LinL2 F_w{velocity_handler, fw};

			F_u.set_global(F.subspan(0,N));
			F_v.set_global(F.subspan(N,N));
			F_w.set_global(F.subspan(2*N,N));

			using Kernel_type = Kernel<4,LinL2,LinL2,LinL2>;
			Kernel_type kernel(F_u,F_v,F_w);

			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				F_u.set_basis(v_dofs);
				F_v.set_basis(v_dofs);
				F_w.set_basis(v_dofs);
				kernel.dispatch_all();
			};

			//perform the loop
			//TODO: parallel by element color?
			mesh.for_each_active_element(action);
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
			using Form_t = BilinH1_t;
			Form_t A_form(velocity_handler,velocity_handler);
			A_form.set_global(KU.subspan(0,N),   U.subspan(0,N));
			A_form.set_global(KU.subspan(N,N),   U.subspan(N,N));
			A_form.set_global(KU.subspan(2*N,N), U.subspan(2*N,N));

			using Kernel_type = Kernel<4, Form_t>;
			Kernel_type kernel(A_form);

			//set up integrating action over each element
			//TODO: is having three forms in parallel better?
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				A_form.set_basis(v_dofs, v_dofs);
				kernel.dispatch_all();
			};

			//perform the loop
			//TODO: parallel by element color?
			mesh.for_each_active_element(action);

			//apply the boundary conditions to treat each corresponding row as an identity row
			u_bc.apply_matvec(KU.subspan(0,N),   U.subspan(0,N));
			v_bc.apply_matvec(KU.subspan(N,N),   U.subspan(N,N));
			w_bc.apply_matvec(KU.subspan(2*N,N), U.subspan(2*N,N));
		}
		
		//compute some number of iterations of the jacobi iteration on the block diagonal terms
		void jacobi_precondition(std::span<double> x, std::span<double> x_old, std::span<const double> rhs, const int n_steps=1) const {
			assert(x.data() != x_old.data()); //the jacobi method cannot be done in-place
			assert(n_vel_total() == 3*n_vel_individual());
			assert(x.size() == n_vel_total()+n_pres());
			assert(x.size() == x_old.size());
			assert(x.size() == rhs.size());

			//split x and x_old into u,v,w,p components
			const auto N1 = n_vel_individual();
			const auto N2 = n_pres();

			std::span<double> u  = x.subspan(0, N1);
			std::span<double> u0 = x_old.subspan(0, N1);
			std::span<const double> fu = rhs.subspan(0, N1);

			std::span<double> v  = x.subspan(N1, N1);
			std::span<double> v0 = x_old.subspan(N1, N1);
			std::span<const double> fv = rhs.subspan(N1, N1);

			std::span<double> w  = x.subspan(2*N1, N1);
			std::span<double> w0 = x_old.subspan(2*N1, N1);
			std::span<const double> fw = rhs.subspan(2*N1, N1);

			std::span<double> p  = x.subspan(3*N1, N2);
			std::span<double> p0 = x_old.subspan(3*N1, N2);
			std::span<const double> h = rhs.subspan(3*N1, N2);

			//allocate memory to store the diagonal part of the matrix
			std::vector<double> D(N1+N2, 0.0);
			std::span<double> d_v = as_span(D, 0, N1);
			std::span<double> d_p = as_span(D, N1, N2);

			//initialize forms and link to storage
			using A_Form_t = BilinH1Jacobi;
			using M_Form_t = BilinL2Jacobi;
			A_Form_t A_form(velocity_handler, velocity_handler);
			M_Form_t M_form(pressure_handler, pressure_handler);

			A_form.set_global(u,u0,fu);
			A_form.set_global(v,v0,fv);
			A_form.set_global(w,w0,fw);
			A_form.set_global(d_v);

			M_form.set_global(p,p0,h);
			M_form.set_global(d_p);

			//create kernel for the integration and link to the forms
			using Kernel_type = Kernel<4, A_Form_t, M_Form_t>;
			Kernel_type kernel(A_form, M_form);

			//set up integrating action over each element
			auto action = [&](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				const auto p_dofs = pressure_handler.basis_active(el);
				A_form.set_basis(v_dofs, v_dofs);
				M_form.set_basis(p_dofs, p_dofs);
				kernel.dispatch_all();
			};

			//perform the loop
			//TODO: parallel by element color?
			for (int n=0; n<n_steps; ++n) {
				//update guess
				if (n>0) {std::swap(x,x_old);}

				//ensure x is zeroed
				std::fill(x.begin(), x.end(), 0.0);

				//compute D and rhs-(L+U)*
				mesh.for_each_active_element(action);

				//finalize inverse
				A_form.finalize_inverse();
				M_form.finalize_inverse();

				//apply bc
				apply_velocity_bc(x.subspan(0,3*N1));
				apply_pressure_bc(x.subspan(3*N1,N2));
			}
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
			using Form0_type = BilinHdiv_t<0>;
			using Form1_type = BilinHdiv_t<1>;
			using Form2_type = BilinHdiv_t<2>;
			using Kernel_type = Kernel<4, Form0_type, Form1_type, Form2_type>;
			Form0_type  Bx_form(velocity_handler, pressure_handler);
			Form1_type  By_form(velocity_handler, pressure_handler);
			Form2_type  Bz_form(velocity_handler, pressure_handler);
			Kernel_type kernel(Bx_form, By_form, Bz_form);

			//assign subspans to each form
			Bx_form.set_global(GP.subspan(0,N),   P);
			By_form.set_global(GP.subspan(N,N),   P);
			Bz_form.set_global(GP.subspan(2*N,N), P);

			//define integration action
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				const auto p_dofs = pressure_handler.basis_active(el);

				Bx_form.set_basis(v_dofs, p_dofs);
				By_form.set_basis(v_dofs, p_dofs);
				Bz_form.set_basis(v_dofs, p_dofs);
				
				kernel.dispatch_all();
			};

			//perform the loop
			//TODO: parallel by element color?
			mesh.for_each_active_element(action);
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
			using Form0_type = BilinHdivAdj_t<0>;
			using Form1_type = BilinHdivAdj_t<1>;
			using Form2_type = BilinHdivAdj_t<2>;
			using Kernel_type = Kernel<4, Form0_type, Form1_type, Form2_type>;
			Form0_type  Bx_t_form(pressure_handler, velocity_handler);
			Form1_type  By_t_form(pressure_handler, velocity_handler);
			Form2_type  Bz_t_form(pressure_handler, velocity_handler);
			Kernel_type kernel(Bx_t_form, By_t_form, Bz_t_form);

			//assign subspans to each form
			Bx_t_form.set_global(GTU, U.subspan(0,N));
			By_t_form.set_global(GTU, U.subspan(N,N));
			Bz_t_form.set_global(GTU, U.subspan(2*N,N));

			//define integration action
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				const auto p_dofs = pressure_handler.basis_active(el);

				Bx_t_form.set_basis(p_dofs, v_dofs);
				By_t_form.set_basis(p_dofs, v_dofs);
				Bz_t_form.set_basis(p_dofs, v_dofs);
				
				kernel.dispatch_all();
			};

			//perform the loop
			//TODO: parallel by element color?
			mesh.for_each_active_element(action);
		}

		//compute M*P (pressure component of the block diagonal preconditioner)
		void M_P(std::span<double> MP, std::span<const double> P) const {
			assert(MP.size()>0);
			assert(MP.size() == P.size());
			assert(MP.size() == pressure_handler.n_dofs());

			using Form_t   = BilinL2_t;
			using Kernel_type = Kernel<4, Form_t>;
			Form_t   M_form(pressure_handler,pressure_handler);
			M_form.set_global(MP,P);
			Kernel_type kernel(M_form);

			//set up integrating action over each element
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto p_dofs = pressure_handler.basis_active(el);
				M_form.set_basis(p_dofs, p_dofs);
				kernel.dispatch_all();
			};

			//perform the loop
			//TODO: parallel by element color?
			mesh.for_each_active_element(action);
		}

		//compute some number of iterations of Gauss-Seidel (forwards or backwards) on M*P=H
		//using the fact that K is block-diagonal with the same matrix for each block
		template<bool FORWARD=true>
		void M_inv_gs(std::span<double> P, std::span<const double> H, const int n_steps=1) const {
			assert(P.size()>0);
			assert(P.size() == H.size());
			assert(P.size() == pressure_handler.n_dofs());

			//get indices for subspans for u, v, w components
			using Form_t   = BilinL2_t; //TODO: change to jacobi or similar
			using Kernel_type = Kernel<4, Form_t>;
			Form_t M_form(pressure_handler,pressure_handler);
			M_form.set_global(P, H);
			Kernel_type kernel(M_form);

			//set up integrating action over each element
			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto p_dofs = pressure_handler.basis_active(el);
				M_form.set_basis(p_dofs, p_dofs);
				kernel.dispatch_all();
			};

			//perform the loop
			//TODO: parallel by element color?
			for (int n=0; n<n_steps; ++n) {
				mesh.for_each_active_element(action);
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

		void apply_pressure_bc(std::span<double> P) const {
			P[0] = 0;
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


		//solve a few iterations with Eigen
		//pass the rhs explicitly so this can be used with either AMR (rhs is from problem)
		//or multigrid (rhs is residual)
		template<int N_INNER=2>
		void smooth(int n_iter, std::span<const double> rhs, double tol=1e-100, bool print_summary=false) {
			LogTime timer{"Stokes::smooth"};

			using Operator = StokesOperator<V_BC,P_BC,MAX_DEPTH>;
			using Preconditioner = StokesPreconditioner<V_BC,P_BC,MAX_DEPTH,N_INNER>;
			// using Preconditioner = Eigen::IdentityPreconditioner;

			Operator op(*this);

			Eigen::GMRES<Operator, Preconditioner> solver;
			solver.setMaxIterations(n_iter);
			solver.setTolerance(tol);
			solver.compute(op);

			//wrap data into Eigen::VectorXd
			assert(rhs.size()==X.size());
			assert(3*velocity_handler.n_dofs()+pressure_handler.n_dofs()==X.size());

			Eigen::Map<Eigen::VectorXd> X_map(X.data(), X.size());
			const Eigen::Map<const Eigen::VectorXd> R_map(rhs.data(), rhs.size());

			X_map = solver.solveWithGuess(R_map, X_map).eval();

			if (print_summary) {
				std::cout << "iterations: " << solver.iterations() << "\n";
				std::cout << "residual_relative: " << solver.error() << "\n";
				switch (solver.info()) {
					case Eigen::Success: std::cout << "info: Success\n"; break;
					case Eigen::NumericalIssue: std::cout << "info: NumericalIssue\n"; break;
					case Eigen::NoConvergence: std::cout << "info: NoConvergence\n"; break;
					case Eigen::InvalidInput: std::cout << "info: InvalidInput\n"; break;
					default : std::cout << "info: Unknown\n"; break;
				}
			}

		}

		void check_stokes_op() {
			using Operator = StokesOperator<V_BC,P_BC,MAX_DEPTH>;
			Operator op(*this);

			Eigen::VectorXd ones = Eigen::VectorXd::Ones(X.size());
			std::cout << "1*OP*1 = " << ones.dot(op*ones) << std::endl;
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
			auto u_vals = velocity_handler.interpolate_to_vertices(u(), n_verts);
			auto v_vals = velocity_handler.interpolate_to_vertices(v(), n_verts);
			auto w_vals = velocity_handler.interpolate_to_vertices(w(), n_verts);
			auto p_vals = pressure_handler.interpolate_to_vertices(p(), n_verts);

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

