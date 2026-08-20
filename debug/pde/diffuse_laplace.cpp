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
/// Dirichlet Boundary Conditions", arXiv:2509.25115), derived from the
/// weak form of the Poisson problem:
///
///     a(u,v) = int_B  phi*grad(u).grad(v) + eps^-3*(1-phi)*u*v  dx
///     l(v)   = int_B  phi*f*v             + eps^-3*(1-phi)*g*v  dx
///
/// phi is the phase field (~1 outside the spheres -- the true domain,
/// ~0 inside them). f and g are supplied as plain functions of (x,y,z),
/// swappable at the top of main() with no other code changes.
///
/// NOTE: g is evaluated directly at each quadrature point rather than
/// via the paper's full normal-direction extension (their Definition 3).
/// This is acceptable for now -- (1-phi) suppresses this term's weight
/// far from the interface -- but is a known simplification worth
/// revisiting once the base machinery is verified working.
///////////////////////////////////////////////////////////////////////////

using Scalar_t       = double;
using Point_t        = gutil::Point<3,Scalar_t>;
using Box_t          = gutil::Box<3,Scalar_t>;
using Assembly_t     = GV::SignedDistanceSpheres<Scalar_t,GV_TEST_DOMAIN_PERIOD>;
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
	std::string file_name     = "spheres.txt";
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
/// f(x,y,z) and g(x,y,z) -- swap these to change the problem being solved.
/// Defaults: f=1 (simple constant source), g=0 (homogeneous Dirichlet,
/// matching the originally-requested problem).
///////////////////////////////////////////////////////////////////////////
inline constexpr auto f_source = [](Scalar_t /*x*/, Scalar_t /*y*/, Scalar_t /*z*/) constexpr noexcept {
	return Scalar_t{1};
};
inline constexpr auto g_boundary = [](Scalar_t /*x*/, Scalar_t /*y*/, Scalar_t /*z*/) constexpr noexcept {
	return Scalar_t{0};
};

///////////////////////////////////////////////////////////////////////////
/// Weight types -- phi itself, the eps^-3*(1-phi) penalty, and the two
/// RHS weights phi*f and eps^-3*(1-phi)*g. All read the phase field from
/// a shared, statically-configured assembly/eps -- see configure() below.
///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////
/// Form aliases
///////////////////////////////////////////////////////////////////////////
using InteriorWeight_t	= GV::AssemblyPhaseFieldWeight<true,Assembly_t>;
using ExteriorWeight_t	= GV::AssemblyPhaseFieldWeight<false,Assembly_t>;

using LaplaceKernel_t  = GV::H1BilinearKernel<true,true>;
using PenaltyKernel_t  = GV::L2BilinearKernel<true,true>;

using LaplaceForm_t  = GV::BilinearForm<4, Scalar_t, DofHandler_t, DofHandler_t, LaplaceKernel_t, LaplaceWeight_t>;
using PenaltyForm_t  = GV::BilinearForm<4, Scalar_t, DofHandler_t, DofHandler_t, PenaltyKernel_t, PenaltyWeight_t>;
using SourceForm_t   = GV::LinearForm<4, Scalar_t, DofHandler_t, GV::L2LinearKernel<true>, SourceWeight_t>;
using BoundaryForm_t = GV::LinearForm<4, Scalar_t, DofHandler_t, GV::L2LinearKernel<true>, BoundaryWeight_t>;

///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	GUTIL_TIMER("Diffuse domain Laplace solver");

	auto cfg = parse_args(argc, argv);

	// ---- Build mesh + assembly (no refinement yet -- happens per-level below) ----
	Box_t domain{{-1,-1,-1}, {1,1,1}};
	MeshHandler_t mesh_handler(domain, cfg.initial_depth + cfg.n_refine);
	mesh_handler.assembly.push_back(gutil::Sphere<3,Scalar_t>(0.5,{0,0,0}));
	// mesh_handler.build_assembly(cfg.file_name);
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
			// refine near the interface, then prolong the previous solution
			// onto the new, finer dof set -- this becomes the initial guess
			const Scalar_t tol = mesh_handler.min_element_size() + cfg.tol;
			const Mesh_t& mesh = mesh_handler.mesh;

			mesh.request_refine([tol, &mesh_handler, &mesh](Elem_t el) {
				Point_t el_center = mesh.geo_center(el);
				Scalar_t sdf = mesh_handler.assembly.signed_distance(el_center);
				return std::abs(sdf) < tol;
			});
			mesh_handler.mesh.process_refine([](Elem_t) { return true; });

			// NOTE: prolong_coefs() replaces coefs[0]'s entire underlying
			// storage (coefs = std::move(new_coefs)), so any Eigen::Map bound
			// to the OLD storage from a previous iteration is now dangling --
			// a fresh Map must be constructed below, after this call, every
			// single level. Never hold a Map across this call.
			c_handler.prolong_coefs();
		}

		const size_t n = d_handler.n_dofs();
		const Scalar_t eps = 0.666666 * mesh_handler.min_element_size();
		GUTIL_LOG("level ", (int)level, ": n_dofs = ", n, "   eps = ", eps);

		// ---- Configure the weight types for this level's eps ----
		LaplaceWeight_t::configure(mesh_handler.assembly, eps);
		PenaltyWeight_t::configure(mesh_handler.assembly, eps);
		SourceWeight_t::configure(mesh_handler.assembly, eps);
		BoundaryWeight_t::configure(mesh_handler.assembly, eps);

		// ---- Assemble A = Laplace-term + Penalty-term, via shared triplets ----
		LaplaceForm_t laplace_form(d_handler);
		PenaltyForm_t penalty_form(d_handler);

		std::vector<Eigen::Triplet<Scalar_t>> triplets;
		{
			GUTIL_TIMER("Assembling A (triplets)");
			laplace_form.build_triplets(triplets);
			penalty_form.build_triplets(triplets);
		}

		Eigen::SparseMatrix<Scalar_t> A(n, n);
		A.setFromTriplets(triplets.begin(), triplets.end());

		// ---- Assemble b = Source-term + Boundary-term ----
		SourceForm_t   source_form(d_handler);
		BoundaryForm_t boundary_form(d_handler);

		std::vector<Scalar_t> rhs(n, 0);
		{
			GUTIL_TIMER("Assembling b");
			source_form.evaluate_vector(GV::as_span(rhs));
			boundary_form.evaluate_vector(GV::as_span(rhs));
		}
		Eigen::Map<Eigen::VectorXd> b(rhs.data(), n);

		// ---- Solve iteratively, using the (prolonged) coefficients as the ----
		// ---- initial guess. A is symmetric positive-definite (the penalty ----
		// term is strictly positive wherever phi<1, removing the constant ----
		// null-space the H1 term alone would have -- Theorem 2's coercivity ----
		// proof for DDM1) -- so Conjugate Gradient is the appropriate choice. ----
		Eigen::Map<Eigen::VectorXd> x(c_handler.get_coefs(0).data(), c_handler.get_coefs(0).size());
		{
			GUTIL_TIMER("Solving (CG)");
			Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar_t>> solver;
			solver.compute(A);
			if (solver.info() != Eigen::Success) {
				GUTIL_LOG("ERROR: matrix factorization failed at level ", (int)level);
				return 1;
			}
			// two-step form, deliberately not `x = solver.solveWithGuess(b, x);` --
			// reads x once as the guess into a genuinely separate result vector,
			// then assigns, so correctness doesn't depend on exactly how/when
			// solveWithGuess reads its guess argument relative to any aliasing.
			Eigen::VectorXd solved = solver.solveWithGuess(b, x);
			x = solved;

			GUTIL_LOG("  CG iterations: ", solver.iterations(), "   estimated error: ", solver.error());
			if (solver.info() != Eigen::Success) {
				GUTIL_LOG("WARNING: CG did not converge at level ", (int)level);
			}
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

			// c_handler.evaluate() correctly walks the hierarchical basis
			// (summing every active ancestor/descendant dof's contribution at
			// each vertex) -- reading coefficients per-vertex directly would
			// silently give the wrong physical-space values.
			std::vector<Scalar_t> u_vals = c_handler.evaluate(0, mesh.vertex_begin(), mesh.vertex_end());
			auto u_lookup = GV::make_index_lookup<float>(
				[&](uint64_t idx) { return (float)u_vals[idx]; }, "u");

			mesh.append_point_data_field_binary(filename, "point_data", sd_lookup, phi_lookup, u_lookup);
		}
	}

	return 0;
}