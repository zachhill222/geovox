#pragma once

#include "fem/fem.hpp"
#include "mesh/mesh.hpp"
#include "simd_keys/simd_keys.hpp"

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>


namespace GV {
	

	/////////////////////////////////////////////////////////////
	/// A class to organize a diffuse stokes problem using
	/// Q1 dofs for pressure and Q1-iso-Q2 dofs for velocity.
	/// To implement the Q1-iso-Q2 dofs, we use the pressure dofs
	/// as the primary dofs and then uniformly refine them to get
	/// the velocity dofs.
	///
	/// Note that the velocity dofs are periodic while the pressure
	/// dofs are not.
	/////////////////////////////////////////////////////////////
	template<typename AssemblyType, int nQuadPoints=3>
	struct DiffuseStokes {
		

		/////////////////////////////////////////////////////////
		/// Aliases and constants
		/////////////////////////////////////////////////////////
		using Assembly_t    = AssemblyType;
		using MeshHandler_t = DiffuseDomainMeshHandler<Assembly_t>;
		using Scalar_t      = typename Assembly_t::Scalar_t;
		using Point_t       = typename Assembly_t::Point_t;
		using Box_t 		= typename Assembly_t::Box_t;

		using Mesh_t        = typename MeshHandler_t::Mesh_t;
		using Elem_t        = typename Mesh_t::Elem_t;
		using Vert_t 		= typename Mesh_t::Vert_t;

		using P_DOF_t       = Keys::DOFS::VoxelQ1<0b000>;	//non-periodic
		using U_DOF_t       = Keys::DOFS::VoxelQ1<0b111>;	//fully-periodic

		using P_Handler_t   = DofHandler<Mesh_t, P_DOF_t>;
		using U_Handler_t   = DofHandler<Mesh_t, U_DOF_t>;

		using P_Coefs_t     = CoefHandler<P_Handler_t, Scalar_t, 1>;
		using U_Coefs_t     = CoefHandler<U_Handler_t, Scalar_t, 3>;

		using InteriorWeight_t = AssemblyPhaseFieldWeight<true, Assembly_t>;
		using ExteriorWeight_t = AssemblyPhaseFieldWeight<false, Assembly_t>;

		using EigenVec       = Eigen::Matrix<Scalar_t, Eigen::Dynamic, 1>;
		using EigenSpMat	 = Eigen::SparseMatrix<Scalar_t, Eigen::RowMajor>;
		using EigenCompMat	 = Eigen::Matrix<Scalar_t, Eigen::Dynamic, 3>;	//aid solving AX=B once vs. three Ax=b problems
		using Triplet_t    	 = Triplet<Scalar_t,EigenSpMat::StorageIndex,Eigen::RowMajor>;

		using Preconditioner = DiagonalPreconditioner<std::vector<Scalar_t>>;
		using InnerSolver    = Eigen::GMRES<EigenSpMat, Preconditioner>;

		static constexpr int N_QUAD = nQuadPoints;


		/////////////////////////////////////////////////////////
		/// Constructors
		/////////////////////////////////////////////////////////
		explicit DiffuseStokes(const Box_t& domain, uint8_t max_depth) : 
			m_handler(domain, max_depth),
			p_handler(m_handler.mesh),
			u_handler(m_handler.mesh),
			p_coefs(p_handler),
			u_coefs(u_handler) {}


		/////////////////////////////////////////////////////////
		/// Problem data
		/////////////////////////////////////////////////////////
		Point_t          body_acceleration{1,0,0};
		Scalar_t         viscosity{1};


		/////////////////////////////////////////////////////////
		/// Primary data
		/////////////////////////////////////////////////////////
		MeshHandler_t m_handler;	//owns the mesh
		P_Handler_t   p_handler;
		U_Handler_t   u_handler;
		P_Coefs_t 	  p_coefs;
		U_Coefs_t     u_coefs;


		/////////////////////////////////////////////////////////
		/// Extra data for (bi)linear forms
		/////////////////////////////////////////////////////////
		InteriorWeight_t interior_wt{};	//phi
		ExteriorWeight_t exterior_wt{};	//1-phi
		Scalar_t		 eps_scale{1};	//eps = min_h * eps_scale


		/////////////////////////////////////////////////////////
		/// Iteration data
		/////////////////////////////////////////////////////////
		struct IterationData {
			Scalar_t abs_tol, rel_tol;
			int max_iterations;

			IterationData(Scalar_t atol, Scalar rtol, int miter) :
				abs_tol{atol}, rel_tol{rtol}, max_iterations{miter} {}
		};

		IterationData inner_iter{1e-6, 1e-6, 100};	//e.g., CG or GMRES
		IterationData outer_iter{1e-8, 1e-8, 50};	//e.g., Uzawa
		Scalar_t 	  relax_w{1};					//Uzawa relxation parameter


		/////////////////////////////////////////////////////////
		/// Utility methods
		/////////////////////////////////////////////////////////
		void initialize(uint8_t initial_depth) noexcept {
			m_handler.mesh.set_depth(initial_depth);
			p_handler.init_dofs();

			set_velocity_dofs();
			p_coefs.init_coefs();	//set pressure to 0
			u_coefs.init_coefs();	//set each velocity component to 0
		}

		void set_velocity_dofs() {
			//set the velocity dofs to be the geometric children of the pressure dofs
			//this makes the velocity dofs Q1-iso-Q2 relative to the pressure so that
			//the inf-sup condition is satisfied. Additionally, refine the mesh so that
			//the quadrature happens on the finer velocity support elements.

			u_handler.clear();
			for (auto it=p_handler.active_dofs.begin(); it!=p_handler.active_dofs.end(); ++it) {
				U_DOF_t u_dof = static_cast<U_DOF_t>(*it);
				if (!u_dof.exists()) {continue;}

				for (U_DOF_t dof : u_dof.children()) {
					if (!dof.exists()) {continue;}
					u_handler.activate(dof);
				}
			}
			u_handler.collect_dofs();

			//the velocity dofs need finer quadrature elements
			m_handler.mesh.process_refine();
		}

		void geometry_refine(Scalar_t g_tol) {
			//refine pressure, update velocity
			Scalar_t tol_dist = g_tol * m_handler.min_element_size();
			std::vector<Elem_t> elems = m_handler.mesh.select_elements(
				[this,tol_dist](Elem_t el) {
					const auto pt = m_handler.mesh.geo_center(el);
					return gutil::abs(m_handler.assembly.signed_distance(pt)) < tol_dist;
				});

			p_handler.refine_quasi_hierarchical(elems);
			set_velocity_dofs();

			p_coefs.prolong_coefs();
			u_coefs.prolong_coefs();
		}


		/////////////////////////////////////////////////////////
		/// Factories to make the (bi)linear forms. Be sure to set eps first.
		/////////////////////////////////////////////////////////
		auto make_A_form() const {
			//a(u_i,v) = int( grad(u_i)*grad(v) * mu*phi)
			return viscosity * MakeBilinearForm<N_QUAD,Scalar_t>(u_handler, H1BilinearKernel_SW{}, InteriorWeight_t{});
		}

		auto make_A_penalty_form() const {
			//a_p(u_i,v) = int(u_i * v * mu*(1-phi)/eps^3)
			const Scalar_t eps = ExteriorWeight_t::eps;	//static, same as InteriorWeight_t::eps
			return (viscosity/(eps*eps*eps)) * MakeBilinearForm<N_QUAD,Scalar_t>(u_handler, L2BilinearKernel_SW{}, ExteriorWeight_t{});
		}

		auto make_M_velocity_form() const {
			//m(u,v) = int (u*v*phi) for use in computing L2 norms
			return MakeBilinearForm<N_QUAD,Scalar_t>(u_handler,L2BilinearKernel_SW{}, InteriorWeight_t{});
		}

		auto make_M_pressure_form() const {
			//m(p,q) = int (p*q*phi) for use in computing L2 norms
			return MakeBilinearForm<N_QUAD,Scalar_t>(p_handler,L2BilinearKernel_SW{}, InteriorWeight_t{});
		}

		template<int Axis> requires(0<=Axis && Axis<3)
		auto make_G_form() const {
			//G(p,v) = int( \partial_i(p) * v * phi)
			//(Top right block, p is trial, v is test)
			return MakeBilinearForm<N_QUAD,Scalar_t>(p_handler, u_handler, DivComponentBilinearKernel<Axis,true>{}, InteriorWeight_t{});
		}

		template<int Axis> requires(0<=Axis && Axis<3)
		auto make_Gt_form() const {
			//Gt(u,q) = int( \partial_i(u) * q * phi)
			//(Bottom left block, u is trial, q is test)
			return MakeBilinearForm<N_QUAD,Scalar_t>(u_handler, p_handler, DivComponentBilinearKernel<Axis,true>{}, InteriorWeight_t{});
		}

		template<int Axis> requires(0<=Axis && Axis<3)
		auto make_acc_form() const {
			//l(v) = int( f_i * v * phi) where f_i is the axis component of the body acceleration
			return MakeLinearForm<N_QUAD,Scalar_t>(u_handler, L2LinearKernel_W{}, body_acceleration[Axis]*InteriorWeight_t{});
		}
		
		/////////////////////////////////////////////////////////
		/// Methods to check for convergence
		/////////////////////////////////////////////////////////
		[[nodiscard]] Scalar_t Rdiv(const EigenSpMat& Mp, std::span<const Scalar_t> u, std::span<const Scalar_t> v, std::span<const Scalar_t> w) const noexcept {
			//compute ||div(velocity)||_L2 / ||velocity||_L2 via integral L2 norms weighted by the diffuse domain heaviside function
			GUTIL_ASSERT(u.size()==v.size() && v.size()==w.size() && w.size()==u_handler.n_dofs());
			GUTIL_ASSERT(Mp.rows()==Mp.cols() && Mp.cols()==p_handler.n_dofs());

			//compute the velocity divergence field at the pressure dofs
			//then recover the coefficients for the divergence field
			EigenVec v_div = EigenVec::Zero(p_handler.n_dofs());
			
			auto Gt_form0 = make_Gt_form<0>(); Gt_form0.mat_vec_multiply_accumulate(GV::as_span(v_div), u);
			auto Gt_form1 = make_Gt_form<1>(); Gt_form1.mat_vec_multiply_accumulate(GV::as_span(v_div), v);
			auto Gt_form2 = make_Gt_form<2>(); Gt_form2.mat_vec_multiply_accumulate(GV::as_span(v_div), w);

			//compute the L2 norm of the divergence
			Eigen::ConjugateGradient<EigenSpMat, Eigen::Lower|Eigen::Upper, Preconditioner> solver;
			solver.compute(Mp);
			EigenVec v_div_coef = solver.solve(v_div);		//M*v_div_coef = Gt*V = Gt0*u + Gt1*v + Gt2*w
			Scalar_t div_norm2  = v_div_coef.dot(v_div);	//v_div_coef^t * M * v_div_coef = v_div_coef^t * v_div

			//compute the L2 norm of the velocity
			auto Mv = make_M_velocity_form();
			Scalar_t v_norm2 = Mv.evaluate_quadratic_form(u);
			v_norm2 += Mv.evaluate_quadratic_form(v);
			v_norm2 += Mv.evaluate_quadratic_form(w);

			return std::sqrt(div_norm2/v_norm2);
		}

		[[nodiscard]] Scalar_t Rdiv(std::span<const Scalar_t> u, std::span<const Scalar_t> v, std::span<const Scalar_t> w) const noexcept {
			//compute ||div(velocity)||_L2 / ||velocity||_L2 via integral L2 norms weighted by the diffuse domain heaviside function
			GUTIL_ASSERT(u.size()==v.size() && v.size()==w.size() && w.size()==u_handler.n_dofs());
			
			auto Mp_form = make_M_pressure_form();
			std::vector<Triplet_t> triplets;
			Mp_form.build_triplets(triplets);
			EigenSpMat Mp(p_handler.n_dofs(), p_handler.n_dofs());
			Mp.setFromTriplets(triplets.begin(), triplets.end());
			return Rdiv(Mp,u,v,w);
		}


		/////////////////////////////////////////////////////////
		/// Inner solvers. The dofs and coefs must be up to date.
		/////////////////////////////////////////////////////////
		// void uzawa_no_project() {
		// 	//set up linear and bilinear forms
		// 	InteriorWeight_t::SetEps(eps_scale * m_handler.min_element_size());
		// 	auto A_form  = make_A_form();			//velocity-velocity block (same for each component)
		// 	auto Ap_form = make_A_penalty_form();	//penalty portion of velocity-velocity block

		// 	auto G_form0  = make_G_form<0>();		//upper-right velocity-pressure block
		// 	auto G_form1  = make_G_form<1>();		//upper-right velocity-pressure block
		// 	auto G_form2  = make_G_form<2>();		//upper-right velocity-pressure block

		// 	auto Gt_form0  = make_Gt_form<0>();		//lower-left pressure-velocity block
		// 	auto Gt_form1  = make_Gt_form<1>();		//lower-left pressure-velocity block
		// 	auto Gt_form2  = make_Gt_form<2>();		//lower-left pressure-velocity block

		// 	auto F_form0  = make_acc_form<0>();		//body acceleration linear form
		// 	auto F_form1  = make_acc_form<1>();		//body acceleration linear form
		// 	auto F_form2  = make_acc_form<2>();		//body acceleration linear form

		// 	//copy the velocity components to a dense column-major matrix
		// 	//so that each component can be solved simultaneously
		// 	const size_t n = u_handler.n_dofs();

		// 	EigenCompMat X(n, 3);
		// 	for (int i=0; i<3; ++i) {
		// 		X.col(i) = Eigen::Map<EigenVec>(u_coefs.get_coefs(i).data(), n); //copies into X
		// 	}

		// 	//build the velocity-velcity matrix
		// 	std::vector<Triplet_t> triplets;
		// 	A_form.build_triplets(triplets);
		// 	Ap_form.build_triplets(triplets);

		// 	EigenSpMat A(n,n);
		// 	A.setFromTriplets(triplets.begin(), triplets.end());
		// 	triplets.clear(); triplets.shrink_to_fit();

		// 	//set up solver
		// 	InnerSolver solver;
		// 	solver.compute(A);
		// 	for (size_t i=0; i<inner_iter.max_iterations; ++i) {
		// 		//set up RHS
		// 		EigenCompMat B = EigenCompMat::Zero(n,3);
		// 		G_form0.mat_vec_multiply_accumulate(std::span<Scalar_t>(B.col(0).data(), n), p_coefs.get_coefs(0))
		// 		if (body_acceleration[0]!=Scalar_t{0}) {
		// 			F_form0.evaluate_vector(std::span<Scalar_t>(B.col(0).data(), n));
		// 		}
		// 		G_form1.mat_vec_multiply_accumulate(std::span<Scalar_t>(B.col(1).data(), n), p_coefs.get_coefs(0))
		// 		if (body_acceleration[1]!=Scalar_t{0}) {
		// 			F_form1.evaluate_vector(std::span<Scalar_t>(B.col(1).data(), n));
		// 		}
		// 		G_form2.mat_vec_multiply_accumulate(std::span<Scalar_t>(B.col(2).data(), n), p_coefs.get_coefs(0))
		// 		if (body_acceleration[2]!=Scalar_t{0}) {
		// 			F_form2.evaluate_vector(std::span<Scalar_t>(B.col(2).data(), n));
		// 		}


		// 		//solve velocity part of uzawa iteration
		// 		X = solver.solveWithGuess(B, X);

		// 		//update pressure part of uzawa iteration
		// 		//TODO: solve via the pressure mass matrix
		// 		Gt_form0.mat_vec_multiply_accumulate(p_coefs.get_coefs(0), u_coefs.get_coefs(0), relax_w);
		// 		Gt_form1.mat_vec_multiply_accumulate(p_coefs.get_coefs(0), u_coefs.get_coefs(1), relax_w);
		// 		Gt_form2.mat_vec_multiply_accumulate(p_coefs.get_coefs(0), u_coefs.get_coefs(2), relax_w);


		// 		//check for convergence
		// 	}

		// 	//copy back to the coefficient handlers
		// 	for (int j=0; j<3; ++j) {
		// 		std::copy(X.col(j).data(), X.col(j).data()+n, u_coefs.get_coefs(j).data());
		// 	}
		// }


		///////////////////////////////////////////////////////////////////////////////////////////
		/// Save current solution as a .vtk file
		///////////////////////////////////////////////////////////////////////////////////////////
		void save_as(const std::string& filename) const {
			//save mesh topology
			m_handler.mesh.collect_vertices();
			m_handler.mesh.save_as_binary(filename);

			//evaluate velocity and pressure fields
			std::vector<Scalar_t> u_vals = u_handler.evaluate(0, m_handler.mesh.vertex_begin(), m_handler.mesh.vertex_end());
			std::vector<Scalar_t> v_vals = u_handler.evaluate(1, m_handler.mesh.vertex_begin(), m_handler.mesh.vertex_end());
			std::vector<Scalar_t> w_vals = u_handler.evaluate(2, m_handler.mesh.vertex_begin(), m_handler.mesh.vertex_end());
			std::vector<Scalar_t> p_vals = p_handler.evaluate(0, m_handler.mesh.vertex_begin(), m_handler.mesh.vertex_end());

			//append solution and diffuse domain heaviside values
			auto velocity_lookup = GV::make_index_lookup<Point_t>(
				[&](uint64_t idx) { return Point_t{u_vals[idx], v_vals[idx], w_vals[idx]}; }, "velocity");

			auto pressure_lookup = GV::make_index_lookup<Scalar_t>(
				[&](uint64_t idx) { return p_vals[idx]; }, "pressure");

			const Scalar_t eps = InteriorWeight_t::eps;
			auto heaviside_lookup = GV::make_feature_lookup<Vert_t>(
				[&, eps](Vert_t vtx) { return m_handler.assembly.heaviside_tanh(m_handler.mesh.geo_coord(vtx), eps);}, "heaviside_tanh");

			m_handler.mesh.append_point_data_field_binary(filename, "solution", velocity_lookup, pressure_lookup, heaviside_lookup);
		}
	};


}