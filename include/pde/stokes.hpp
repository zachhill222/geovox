#pragma once

#include "fem/fem.hpp"
#include "mesh/mesh.hpp"
#include "simd_keys/simd_keys.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>


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
		using Triplet_t    	 = Triplet<Scalar_t, typename EigenSpMat::StorageIndex,Eigen::RowMajor>;

		using DiagPrecon     = DiagonalPreconditioner<std::vector<Scalar_t>>;
		// using BlockSolver    = Eigen::ConjugateGradient<EigenSpMat, Eigen::Upper|Eigen::Lower, DiagPrecon>;
		// using BlockSolver    = Eigen::GMRES<EigenSpMat, DiagPrecon>;
		// using BlockSolver    = Eigen::SimplicialLDLT<EigenSpMat>;
		// using BlockSolver    = Eigen::IncompleteLUT<Scalar_t>;
		using BlockSolver    = Eigen::IncompleteCholesky<Scalar_t>;
		using Preconditioner = StokesBlockDiagonalPreconditioner<EigenSpMat, BlockSolver, BlockSolver>;
		using MonolithicSolver = Eigen::GMRES<EigenSpMat, Preconditioner>;
		// using MonolithicSolver = Eigen::MINRES<EigenSpMat, Eigen::Upper|Eigen::Lower, Preconditioner>;

		using InnerSolver    = Eigen::GMRES<EigenSpMat, DiagPrecon>;
		// using InnerSolver    = Eigen::MINRES<EigenSpMat, Eigen::Upper|Eigen::Lower, DiagPrecon>;
		// using InnerSolver    = Eigen::ConjugateGradient<EigenSpMat, Eigen::Upper|Eigen::Lower, DiagPrecon>;
		// using InnerSolver    = Eigen::SimplicialLDLT<EigenSpMat>;

		static constexpr int N_QUAD = nQuadPoints;


		/////////////////////////////////////////////////////////
		/// Constructors
		/////////////////////////////////////////////////////////
		explicit DiffuseStokes(const Box_t& domain, uint8_t max_depth) : 
			m_handler(domain, max_depth),
			p_handler(m_handler.mesh),
			u_handler(m_handler.mesh),
			p_coefs(p_handler),
			u_coefs(u_handler) {
				InteriorWeight_t::SetAssembly(m_handler.assembly);
			}


		/////////////////////////////////////////////////////////
		/// Problem data
		/////////////////////////////////////////////////////////
		Point_t          body_acceleration{1,0,0};
		Scalar_t         viscosity{1};

		void set_assembly_from_file(const std::string& filename) {
			m_handler.build_assembly(filename);
		}

		void add_unit_sphere() noexcept {
			//TODO: remove if we add other particle shapes
			m_handler.assembly.push_back(typename Assembly_t::Particle_t{Point_t::Filled(Scalar_t{0}),Scalar_t{1}});
		}

		/////////////////////////////////////////////////////////
		/// Primary data
		/////////////////////////////////////////////////////////
		MeshHandler_t m_handler;	//owns the mesh and diffuse_domain logic
		P_Handler_t   p_handler;	//pressure dofs
		U_Handler_t   u_handler;	//velocity dofs (shared for all components)
		P_Coefs_t 	  p_coefs;		//pressure dof coefficients
		U_Coefs_t     u_coefs;		//velocity dof coefficients (one field per component)


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
			int max_iter;

			IterationData(Scalar_t atol, Scalar_t rtol, int miter) :
				abs_tol{atol}, rel_tol{rtol}, max_iter{miter} {}
		};

		IterationData inner_iter{1e-6, 1e-6, 1000};	//e.g., CG or GMRES
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
			{
				auto lock = u_handler.begin_key_mask_unstable();
				for (auto it=p_handler.active_dofs.begin(); it!=p_handler.active_dofs.end(); ++it) {
					U_DOF_t u_dof{it->key};
					if (!u_dof.is_valid()) {continue;}

					for (U_DOF_t dof : u_dof.children()) {
						if (!dof.exists()) {continue;}
						u_handler.activate(dof);
					}
				}
				u_handler.end_key_mask_unstable();
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

		void refine_interior(Scalar_t cutoff = Scalar_t{0.25} ) {
			const Scalar_t eps = read_eps();
			std::vector<Elem_t> elems = m_handler.mesh.select_elements(
				[this, cutoff, eps](Elem_t el) {
					const auto pt = m_handler.mesh.geo_center(el);
					return m_handler.assembly.heaviside(pt, eps) > cutoff;
				});
			p_handler.refine_quasi_hierarchical(elems);
			set_velocity_dofs();

			p_coefs.prolong_coefs();
			u_coefs.prolong_coefs();
		}

		void update_eps() noexcept {
			InteriorWeight_t::SetEps(eps_scale * m_handler.min_element_size());
		}

		Scalar_t read_eps() noexcept {
			return InteriorWeight_t::eps;
		}

		void pin_pressure() noexcept {
			p_coefs.get_coefs(0)[0] = Scalar_t{0};
		}


		/////////////////////////////////////////////////////////
		/// Factories to make the (bi)linear forms. Be sure to set eps first.
		/////////////////////////////////////////////////////////
		auto make_A_form() const {
			//a(u_i,v) = int( grad(u_i)*grad(v) * mu*phi)
			return MakeBilinearForm<N_QUAD,Scalar_t>(u_handler, H1BilinearKernel_SW{}, viscosity * InteriorWeight_t{});
		}

		auto make_A_penalty_form() const {
			//a_p(u_i,v) = int(u_i * v * mu*(1-phi)/eps^3)
			const Scalar_t eps = ExteriorWeight_t::eps;	//static, same as InteriorWeight_t::eps
			return MakeBilinearForm<N_QUAD,Scalar_t>(u_handler, L2BilinearKernel_SW{}, (viscosity/(eps*eps*eps)) * ExteriorWeight_t{});
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
		/// Factories to build block matrices
		/////////////////////////////////////////////////////////
		auto make_pressure_mass_mat() const {
			auto Mp_form = make_M_pressure_form();
			std::vector<Triplet_t> triplets;
			Mp_form.build_triplets(triplets);
			EigenSpMat Mp(p_handler.n_dofs(), p_handler.n_dofs());
			Mp.setFromTriplets(triplets.begin(), triplets.end());
			return Mp;
		}

		auto make_v_v_block() const {
			auto A_form  = make_A_form();
			auto Ap_form = make_A_penalty_form();
			std::vector<Triplet_t> triplets;
			A_form.build_triplets(triplets);
			Ap_form.build_triplets(triplets);
			
			const size_t n = u_handler.n_dofs();
			EigenSpMat A(n,n);
			A.setFromTriplets(triplets.begin(), triplets.end());
			return A;
		}

		template<int Axis> requires(0<=Axis && Axis<3)
		auto make_v_p_block() const {
			auto G_form = make_G_form<Axis>();
			std::vector<Triplet_t> triplets;
			G_form.build_triplets(triplets);
			
			const size_t n = u_handler.n_dofs();
			const size_t m = p_handler.n_dofs();
			EigenSpMat G(n,m);
			G.setFromTriplets(triplets.begin(), triplets.end());
			return G;
		}

		[[nodiscard]] auto make_monolithic_mat_coo() const {
			const size_t n_v = u_handler.n_dofs();
			
			auto A_form   = make_A_form();
			auto Ap_form  = make_A_penalty_form();
			
			auto G_form0  = make_G_form<0>();
			auto G_form1  = make_G_form<1>();
			auto G_form2  = make_G_form<2>();

			auto Gt_form0 = make_Gt_form<0>();
			auto Gt_form1 = make_Gt_form<1>();
			auto Gt_form2 = make_Gt_form<2>();

			std::vector<Triplet_t> triplets;

			// diagonal A blocks -- same A for each velocity component, offset by c*n_v
			for (size_t c=0; c<3; ++c) {
				A_form.build_triplets(triplets, c*n_v, c*n_v);
				Ap_form.build_triplets(triplets, c*n_v, c*n_v);
			}

			// off-diagonal G blocks (row=velocity component c, col=pressure, starting at 3*n_v)
			G_form0.build_triplets(triplets, 0*n_v, 3*n_v);
			G_form1.build_triplets(triplets, 1*n_v, 3*n_v);
			G_form2.build_triplets(triplets, 2*n_v, 3*n_v);

			// off-diagonal G^t blocks (row=pressure at 3*n_v, col=velocity component c)
			Gt_form0.build_triplets(triplets, 3*n_v, 0*n_v);
			Gt_form1.build_triplets(triplets, 3*n_v, 1*n_v);
			Gt_form2.build_triplets(triplets, 3*n_v, 2*n_v);

			return triplets;
		}

		[[nodiscard]] auto make_monolithic_rhs() const {
			const size_t n_v = u_handler.n_dofs();
			const size_t n_p = p_handler.n_dofs();
			const size_t N = 3*n_v + n_p;

			auto F_form0 = make_acc_form<0>();
			auto F_form1 = make_acc_form<1>();
			auto F_form2 = make_acc_form<2>();

			std::vector<Scalar_t> rhs(N, Scalar_t{0});   // last n_p entries stay 0 -- the incompressibility constraint's own rhs is homogeneous (h=0)

			if (body_acceleration[0]!=Scalar_t{0}) { F_form0.evaluate_vector(std::span<Scalar_t>(rhs.data()+0*n_v, n_v)); }
			if (body_acceleration[1]!=Scalar_t{0}) { F_form1.evaluate_vector(std::span<Scalar_t>(rhs.data()+1*n_v, n_v)); }
			if (body_acceleration[2]!=Scalar_t{0}) { F_form2.evaluate_vector(std::span<Scalar_t>(rhs.data()+2*n_v, n_v)); }

			return rhs;
		}


		/////////////////////////////////////////////////////////
		/// Utility methods to copy velocity to/from an eigen dense matrix
		/////////////////////////////////////////////////////////
		[[nodiscard]] EigenCompMat get_velocity_as_mat() const {
			const size_t n_v = u_handler.n_dofs();
			EigenCompMat X(n_v, 3);
			X.col(0) = Eigen::Map<const EigenVec>(u_coefs.get_coefs(0).data(), n_v); //copy, not a view
			X.col(1) = Eigen::Map<const EigenVec>(u_coefs.get_coefs(1).data(), n_v); //copy, not a view
			X.col(2) = Eigen::Map<const EigenVec>(u_coefs.get_coefs(2).data(), n_v); //copy, not a view
			return X;
		}

		void copy_back_velocity(const EigenCompMat& X) {
			GUTIL_ASSERT(u_handler.n_dofs() == (size_t) X.rows());
			const size_t n_v = u_handler.n_dofs();

			for (int j=0; j<3; ++j) {
				std::copy(X.col(j).data(), X.col(j).data()+n_v, u_coefs.get_coefs(j).data());
			}
		}



		/////////////////////////////////////////////////////////
		/// Utility methods to compute vector products with individual blocks
		/////////////////////////////////////////////////////////
		[[nodiscard]] EigenCompMat apply_G(std::span<const Scalar_t> p) const {
			//column i of the result is the velocity component i result.
			const size_t n_v = u_handler.n_dofs();
			EigenCompMat result = EigenCompMat::Zero(n_v, 3);
			auto g0 = make_G_form<0>(); g0.mat_vec_multiply_accumulate(std::span<Scalar_t>(result.col(0).data(), n_v), p);
			auto g1 = make_G_form<1>(); g1.mat_vec_multiply_accumulate(std::span<Scalar_t>(result.col(1).data(), n_v), p);
			auto g2 = make_G_form<2>(); g2.mat_vec_multiply_accumulate(std::span<Scalar_t>(result.col(2).data(), n_v), p);
			return result;
		}

		[[nodiscard]] EigenVec apply_Gt(std::span<const Scalar_t> u0, std::span<const Scalar_t> u1, std::span<const Scalar_t> u2) const {
			// G^t*u = Gt0*u0 + Gt1*u1 + Gt2*u2
			const size_t n_p = p_handler.n_dofs();
			EigenVec result = EigenVec::Zero(n_p);
			auto gt0 = make_Gt_form<0>(); gt0.mat_vec_multiply_accumulate(std::span<Scalar_t>(result.data(), n_p), u0);
			auto gt1 = make_Gt_form<1>(); gt1.mat_vec_multiply_accumulate(std::span<Scalar_t>(result.data(), n_p), u1);
			auto gt2 = make_Gt_form<2>(); gt2.mat_vec_multiply_accumulate(std::span<Scalar_t>(result.data(), n_p), u2);
			return result;
		}

		[[nodiscard]] EigenVec apply_Gt(const EigenCompMat& u) const {
			const size_t n_v = u_handler.n_dofs();
			return apply_Gt(
				std::span<const Scalar_t>(u.col(0).data(), n_v),
				std::span<const Scalar_t>(u.col(1).data(), n_v),
				std::span<const Scalar_t>(u.col(2).data(), n_v));
		}

		[[nodiscard]] EigenCompMat assemble_momentum_rhs(std::span<const Scalar_t> p) const {
			//for the system K*u = f - G*p
			EigenCompMat result = -apply_G(p);   // G0*p, G1*p, G2*p per column

			auto f0 = make_acc_form<0>();
			auto f1 = make_acc_form<1>();
			auto f2 = make_acc_form<2>();
			const size_t n_v = u_handler.n_dofs();
			if (body_acceleration[0]!=Scalar_t{0}) { f0.evaluate_vector(std::span<Scalar_t>(result.col(0).data(), n_v)); }
			if (body_acceleration[1]!=Scalar_t{0}) { f1.evaluate_vector(std::span<Scalar_t>(result.col(1).data(), n_v)); }
			if (body_acceleration[2]!=Scalar_t{0}) { f2.evaluate_vector(std::span<Scalar_t>(result.col(2).data(), n_v)); }
			return result;
		}


		/////////////////////////////////////////////////////////
		/// Methods to check for convergence
		/////////////////////////////////////////////////////////
		[[nodiscard]] Scalar_t Rdiv(const EigenSpMat& Mp, std::span<const Scalar_t> u, std::span<const Scalar_t> v, std::span<const Scalar_t> w) const noexcept {
			//compute ||div(velocity)||_L2 / ||velocity||_L2 via integral L2 norms weighted by the diffuse domain heaviside function
			GUTIL_ASSERT(u.size()==v.size() && v.size()==w.size() && w.size()==u_handler.n_dofs());
			GUTIL_ASSERT(Mp.rows()==Mp.cols() && (size_t) Mp.cols()==p_handler.n_dofs());

			//compute the velocity divergence field at the pressure dofs
			//then recover the coefficients for the divergence field
			EigenVec v_div = EigenVec::Zero(p_handler.n_dofs());
			
			auto Gt_form0 = make_Gt_form<0>(); Gt_form0.mat_vec_multiply_accumulate(GV::as_span(v_div), u);
			auto Gt_form1 = make_Gt_form<1>(); Gt_form1.mat_vec_multiply_accumulate(GV::as_span(v_div), v);
			auto Gt_form2 = make_Gt_form<2>(); Gt_form2.mat_vec_multiply_accumulate(GV::as_span(v_div), w);

			//compute the L2 norm of the divergence
			Eigen::ConjugateGradient<EigenSpMat, Eigen::Lower|Eigen::Upper, DiagPrecon> solver;
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

		[[nodiscard]] Scalar_t Rdiv(const EigenSpMat& Mp, const EigenCompMat& X) const noexcept {
			GUTIL_ASSERT(u_handler.n_dofs() == (size_t) X.rows());
			const size_t n_v = u_handler.n_dofs();

			return Rdiv(Mp, std::span<const Scalar_t>(X.col(0).data(), n_v), 
							std::span<const Scalar_t>(X.col(1).data(), n_v), 
							std::span<const Scalar_t>(X.col(2).data(), n_v));
		}

		[[nodiscard]] Scalar_t Rdiv(std::span<const Scalar_t> u, std::span<const Scalar_t> v, std::span<const Scalar_t> w) const noexcept {
			//compute ||div(velocity)||_L2 / ||velocity||_L2 via integral L2 norms weighted by the diffuse domain heaviside function
			GUTIL_ASSERT(u.size()==v.size() && v.size()==w.size() && w.size()==u_handler.n_dofs());
			return Rdiv(make_pressure_mass_mat(),u,v,w);
		}

		[[nodiscard]] Scalar_t Rdiv() const noexcept {
			return Rdiv(make_pressure_mass_mat(), u_coefs.get_coefs(0), u_coefs.get_coefs(1), u_coefs.get_coefs(2));
		}


		


		///////////////////////////////////////////////////////////////////////////////////////////
		/// Save current solution as a .vtk file
		///////////////////////////////////////////////////////////////////////////////////////////
		void save_as(const std::string& filename) const {
			//save mesh topology
			m_handler.mesh.collect_vertices();
			m_handler.mesh.save_as_binary(filename);

			//evaluate velocity and pressure fields
			std::vector<Scalar_t> u_vals = u_coefs.evaluate(0, m_handler.mesh.vertex_begin(), m_handler.mesh.vertex_end());
			std::vector<Scalar_t> v_vals = u_coefs.evaluate(1, m_handler.mesh.vertex_begin(), m_handler.mesh.vertex_end());
			std::vector<Scalar_t> w_vals = u_coefs.evaluate(2, m_handler.mesh.vertex_begin(), m_handler.mesh.vertex_end());
			std::vector<Scalar_t> p_vals = p_coefs.evaluate(0, m_handler.mesh.vertex_begin(), m_handler.mesh.vertex_end());

			//append solution and diffuse domain heaviside values
			auto velocity_lookup = GV::make_index_lookup<Point_t>(
				[&](uint64_t idx) { return Point_t{u_vals[idx], v_vals[idx], w_vals[idx]}; }, "velocity");

			auto pressure_lookup = GV::make_index_lookup<Scalar_t>(
				[&](uint64_t idx) { return p_vals[idx]; }, "pressure");

			const Scalar_t eps = InteriorWeight_t::eps;
			auto heaviside_lookup = GV::make_feature_lookup<Vert_t>(
				[&, eps](Vert_t vtx) { return m_handler.assembly.heaviside(m_handler.mesh.geo_coord(vtx), eps);}, "heaviside");

			m_handler.mesh.append_point_data_field_binary(filename, "solution", velocity_lookup, pressure_lookup, heaviside_lookup);
		}
	};


}