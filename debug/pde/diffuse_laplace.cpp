#include "gutil.hpp"
#include "geovox.hpp"

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <iostream>
#include <string>

#ifndef GV_TEST_DOMAIN_PERIOD
	#define GV_TEST_DOMAIN_PERIOD 0b111   // fully periodic box boundary in x, y, z
#endif

///////////////////////////////////////////////////////////////////////////
/// Diffuse Domain solver for the Laplace/Poisson equation:
///     -Delta(u) = f    in the true domain (outside the spheres)
///     u = g            on the diffuse interface (sphere surfaces)
///     periodic         on the outer box boundary
///
/// Uses DDM1 (Method 1, Benfield & Dedner, "Diffuse Domain Methods with
/// Dirichlet Boundary Conditions", arXiv:2509.25115):
///
///     a(u,v) = int_B  phi*grad(u).grad(v) + eps^-3*(1-phi)*u*v  dx
///     l(v)   = int_B  phi*f*v             + eps^-3*(1-phi)*g*v  dx
///
/// Source/Boundary weights are built via genuine ProductKernelWeight
/// composition (AssemblyPhaseFieldWeight * FunctionWeight/
/// ModifiedFunctionKernelWeight) -- both sides of every product below
/// share one assembly/eps via the same (default) StaticID group, so the
/// shared-sdf-cache dispatch in Sum/ProductKernelWeight correctly reuses
/// one sdf computation across each product's two factors.
///////////////////////////////////////////////////////////////////////////

using Scalar_t       = double;
using Point_t        = gutil::Point<3,Scalar_t>;
using Box_t          = gutil::Box<3,Scalar_t>;
using Assembly_t     = GV::SignedDistanceSpheres<Scalar_t,GV_TEST_DOMAIN_PERIOD,true>;
using MeshHandler_t  = GV::DiffuseDomainMeshHandler<Assembly_t>;

using Mesh_t         = typename MeshHandler_t::Mesh_t;
using Elem_t         = typename MeshHandler_t::Mesh_t::Elem_t;
using Vert_t         = typename MeshHandler_t::Mesh_t::Vert_t;

using DOF_t          = GV::Keys::DOFS::VoxelQ1<GV_TEST_DOMAIN_PERIOD>;
using DofHandler_t   = GV::DofHandler<Mesh_t,DOF_t>;
using CoefHandler_t  = GV::CoefHandler<DofHandler_t,Scalar_t,1>;

///////////////////////////////////////////////////////////////////////////
/// Runtime test configuration (from argv)
///////////////////////////////////////////////////////////////////////////
struct TestConfig {
	std::string test_name     = "ddm_laplace";
	std::string file_name     = "sphere.txt";
	size_t      initial_depth = 3;
	Scalar_t    tol           = 0.25;   // signed-distance tolerance for boundary refinement
	uint8_t     n_refine      = 3;
};

TestConfig parse_args(int argc, char* argv[]) {
	TestConfig cfg;
	std::vector<std::string> args(argv, argv+argc);
	for (size_t i=0; i<args.size(); ++i) {
		if      (args[i] == "-name") { cfg.test_name     = args[++i]; }
		else if (args[i] == "-file") { cfg.file_name     = args[++i]; }
		else if (args[i] == "-ID")   { cfg.initial_depth = atoi(args[++i].c_str()); }
		else if (args[i] == "-TOL")  { cfg.tol           = atof(args[++i].c_str()); }
		else if (args[i] == "-NR")   { cfg.n_refine      = atoi(args[++i].c_str()); }
	}
	return cfg;
}

///////////////////////////////////////////////////////////////////////////
// Manufactured solution for a sphere of radius R centered at (cx,cy,cz):
// u = R^2 - r^2, satisfying -Delta(u) = 6 and u=0 on the sphere surface.
// NOTE: assumes a specific sphere geometry -- update to match whatever
// sphere.txt actually contains.
///////////////////////////////////////////////////////////////////////////
inline constexpr Scalar_t sphere_R  = 1;
inline constexpr Scalar_t sphere_cx = 0, sphere_cy = 0, sphere_cz = 0;

inline constexpr auto u_exact = [](Scalar_t x, Scalar_t y, Scalar_t z) constexpr noexcept {
	const Scalar_t dx=x-sphere_cx, dy=y-sphere_cy, dz=z-sphere_cz;
	return sphere_R*sphere_R - (dx*dx+dy*dy+dz*dz);
};
inline constexpr auto f_source = [](Scalar_t, Scalar_t, Scalar_t) constexpr noexcept { return Scalar_t{6}; };
inline constexpr auto g_boundary = [](Scalar_t, Scalar_t, Scalar_t) constexpr noexcept { return Scalar_t{0}; };

using L2ErrorWeight_t = GV::L2DiffuseErrorWeight<Assembly_t, u_exact>;
using ErrorForm_t     = GV::LinearForm<4, Scalar_t, DofHandler_t, GV::L2LinearKernel<>, L2ErrorWeight_t>;

///////////////////////////////////////////////////////////////////////////
/// Weight/kernel/form aliases. StaticID left at its default (0) throughout
/// -- this program solves one PDE on one domain, so all diffuse-domain
/// weights below correctly share one assembly/eps via that single,
/// implicit StaticID group -- including across each product's two factors.
///////////////////////////////////////////////////////////////////////////
using InteriorWeight_t = GV::AssemblyPhaseFieldWeight<true,  Assembly_t>;
using ExteriorWeight_t = GV::AssemblyPhaseFieldWeight<false, Assembly_t>;

using SourceWeight_t   = GV::ProductKernelWeight<InteriorWeight_t, GV::ModifiedFunctionKernelWeight<Assembly_t, f_source>>;
using BoundaryWeight_t = GV::ProductKernelWeight<ExteriorWeight_t, GV::ModifiedFunctionKernelWeight<Assembly_t, g_boundary>>;

using LaplaceKernel_t  = GV::H1BilinearKernel<true,true>;
using PenaltyKernel_t  = GV::L2BilinearKernel<true,true>;

using LaplaceForm_t  = GV::BilinearForm<4, Scalar_t, DofHandler_t, DofHandler_t, LaplaceKernel_t, InteriorWeight_t>;
using SourceForm_t   = GV::LinearForm<4, Scalar_t, DofHandler_t, GV::L2LinearKernel<true>, SourceWeight_t>;
using PenaltyForm_t  = GV::BilinearForm<4, Scalar_t, DofHandler_t, DofHandler_t, PenaltyKernel_t,
                            GV::ScaledKernelWeight<Scalar_t, ExteriorWeight_t>>;
using BoundaryForm_t = GV::LinearForm<4, Scalar_t, DofHandler_t, GV::L2LinearKernel<true>,
                            GV::ScaledKernelWeight<Scalar_t, BoundaryWeight_t>>;



///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	GUTIL_TIMER("Diffuse domain Laplace solver");

	auto cfg = parse_args(argc, argv);

	// ---- Build mesh + assembly (no refinement yet -- happens per-level below) ----
	Box_t domain{{-2,-2,-2}, {2,2,2}};
	MeshHandler_t mesh_handler(domain, cfg.initial_depth + cfg.n_refine);
	mesh_handler.build_assembly(cfg.file_name);
	mesh_handler.mesh.set_depth(cfg.initial_depth);

	// ---- DofHandler / CoefHandler setup, at the initial (coarsest) depth ----
	DofHandler_t d_handler(mesh_handler.mesh);
	d_handler.init_dofs();

	CoefHandler_t c_handler(d_handler);
	c_handler.init_coefs(0, [](DOF_t) { return Scalar_t{0}; });   // start from zero at the coarsest level

	// ---- Solve at each refinement level, using the previous level's ----
	// ---- (prolonged) solution as the initial guess for the next solve ----
	for (uint8_t level=0; level<=cfg.n_refine; ++level) {
		GUTIL_TIMER("Refinement level ", (int)level, "/", (int)cfg.n_refine);

		if (level > 0) {
			// select elements near the interface, refine the dof handler
			// explicitly, then process the mesh and prolong the previous
			// solution onto the new, finer dof set -- this becomes the
			// initial guess for this level's solve.
			const Scalar_t tol = mesh_handler.min_element_size() + cfg.tol;
			const Mesh_t& mesh = mesh_handler.mesh;

			// NOTE: assumes select_elements(pred) returns active elements
			// (e.g. std::vector<Elem_t>) matching pred -- not yet added to
			// the mesh class as of this writing.
			std::vector<Elem_t> ref_elems = mesh.select_elements([tol, &mesh_handler, &mesh](Elem_t el) {
				Scalar_t sdf = mesh_handler.assembly.signed_distance(mesh.geo_center(el));
				return std::abs(sdf) < tol;
			});

			d_handler.refine_quasi_hierarchical(ref_elems);
			c_handler.prolong_coefs();

			mesh_handler.mesh.process_refine([](Elem_t) { return true; });
		}

		const size_t n = d_handler.n_dofs();
		const Scalar_t eps = Scalar_t{1} * mesh_handler.min_element_size();
		GUTIL_LOG("level ", (int)level, ": n_dofs = ", n, "   eps = ", eps);

		// ---- Point every diffuse-domain weight (shared StaticID=0 group) ----
		// ---- at this level's assembly/eps before assembling anything.    ----
		GV::AssemblyDiffuseDomain<Assembly_t,0>::SetAssembly(mesh_handler.assembly);
		GV::AssemblyDiffuseDomain<Assembly_t,0>::SetEps(eps);

		// ---- Assemble A = Laplace-term + Penalty-term, via shared triplets ----
		// mesh_handler.mesh.sort_elements_by_color();
		const Scalar_t eps3_inv = Scalar_t{1}/(eps*eps*eps);
		PenaltyForm_t penalty_form(d_handler, PenaltyKernel_t{}, GV::ScaledKernelWeight<Scalar_t, ExteriorWeight_t>(eps3_inv));
		LaplaceForm_t laplace_form(d_handler);
		// GV::BilinearFormOperator laplace_op(laplace_form);
		// GV::BilinearFormOperator penalty_op(penalty_form);
		// auto lhs_op = laplace_op + penalty_op;
		// using Preconditioner_t = GV::JacobiPreconditioner<decltype(laplace_op+penalty_op), 1>;
		using Preconditioner_t = GV::DiagonalPreconditioner<std::vector<Scalar_t>>;


		using EigenSpMat = Eigen::SparseMatrix<Scalar_t, Eigen::RowMajor>;
		using Triplet    = GV::Triplet<Scalar_t,EigenSpMat::StorageIndex,Eigen::RowMajor>;
		std::vector<Triplet> triplets;
		{
			GUTIL_TIMER("Assembling A (triplets)");
			laplace_form.build_triplets(triplets);
			penalty_form.build_triplets(triplets);
		}

		auto t1 = new gutil::LogTime{"Assembling A (Eigen from triplets)"};
		EigenSpMat A(n, n);
		A.setFromTriplets(triplets.begin(), triplets.end());
		delete t1;

		// ---- Assemble b = Source-term + Boundary-term ----
		SourceForm_t   source_form(d_handler);
		BoundaryForm_t boundary_form(d_handler, GV::L2LinearKernel<true>{}, GV::ScaledKernelWeight<Scalar_t, BoundaryWeight_t>(eps3_inv));

		std::vector<Scalar_t> rhs(n, 0);
		{
			GUTIL_TIMER("Assembling b");
			source_form.evaluate_vector(GV::as_span(rhs));
			boundary_form.evaluate_vector(GV::as_span(rhs));
		}
		Eigen::Map<Eigen::VectorXd> b(rhs.data(), n);

		// ---- Solve iteratively, using the (prolonged) coefficients as the ----
		// ---- initial guess. A is symmetric positive-definite (Theorem 2's ----
		// ---- coercivity proof for DDM1) -- Conjugate Gradient is appropriate. ----
		Eigen::Map<Eigen::VectorXd> x(c_handler.get_coefs(0).data(), c_handler.get_coefs(0).size());
		{
			GUTIL_TIMER("Solving (CG)");
			Eigen::ConjugateGradient<EigenSpMat, Eigen::Lower|Eigen::Upper, Preconditioner_t> solver;
			solver.compute(A);
			
			// Eigen::ConjugateGradient<decltype(lhs_op), Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> solver;
			// Eigen::ConjugateGradient<decltype(lhs_op), Eigen::Lower|Eigen::Upper, Preconditioner_t> solver;
			// solver.compute(lhs_op);

			if (solver.info() != Eigen::Success) {
				GUTIL_LOG("ERROR: matrix factorization failed at level ", (int)level);
				return 1;
			}
			Eigen::VectorXd solved = solver.solveWithGuess(b, x);
			x = solved;

			GUTIL_LOG("  CG iterations: ", solver.iterations(), "   estimated error: ", solver.error());
			if (solver.info() != Eigen::Success) {
				GUTIL_LOG("WARNING: CG did not converge at level ", (int)level);
			}
			
			//compute the L2 error
			ErrorForm_t e_form(d_handler);
			auto l2_err = gutil::sqrt(e_form.integrate_weight(c_handler.get_coefs(0)));
			GUTIL_LOG("Computed L2 error: ", l2_err, " n_dofs= ", d_handler.n_dofs(), " min_h= ", mesh_handler.min_element_size());
		}

		// ---- Save a VTK snapshot for this level ----
		{
			const std::string filename = cfg.test_name + "_level" + std::to_string(level) + ".vtk";
			GUTIL_TIMER("Saving mesh as ", filename);

			mesh_handler.mesh.collect_vertices();
			mesh_handler.mesh.save_as_binary(filename);

			const Assembly_t& assembly = mesh_handler.assembly;
			const Mesh_t& mesh = mesh_handler.mesh;

			auto sd_lookup = GV::make_feature_lookup<Vert_t>(
				[&](Vert_t vtx) { return assembly.signed_distance(mesh.geo_coord(vtx)); }, "signed_distance");

			auto phi_lookup = GV::make_feature_lookup<Vert_t>(
				[&](Vert_t vtx) { return assembly.heaviside_tanh(mesh.geo_coord(vtx), eps); }, "phi");

			std::vector<Scalar_t> u_vals = c_handler.evaluate(0, mesh.vertex_begin(), mesh.vertex_end());
			auto u_lookup = GV::make_index_lookup<float>(
				[&](uint64_t idx) { return (float)u_vals[idx]; }, "u");

			mesh.append_point_data_field_binary(filename, "point_data", sd_lookup, phi_lookup, u_lookup);
		}

		mesh_handler.mesh.sort_elements_by_depth();
	}

	return 0;
}