#include "gutil.hpp"
#include "geovox.hpp"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>

///////////////////////////////////////////////////////////////////////////
/// Periodic screened-Poisson / modified Helmholtz example:
///     -Delta(u) + alpha*u = f      on [-1,1]^3,  alpha > 0
/// with fully periodic boundary conditions in x, y, z.
///
/// Manufactured solution:  u(x,y,z) = cos(pi*x)*cos(pi*y)*cos(pi*z)
///   - cos(pi*(-1)) == cos(pi*(1)) == -1, so u is genuinely periodic
///     on this domain (matches at both ends in every axis).
///   - Delta(u) = -3*pi^2 * u, so u is an eigenfunction of -Delta with
///     eigenvalue lambda = 3*pi^2.
///   - Substituting into the strong form gives the closed-form forcing:
///         f = (3*pi^2 + alpha) * u
///
/// Unlike genuine Helmholtz (-Delta(u) - k^2*u = f), this operator's
/// eigenvalues are lambda_n + alpha for every eigenvalue lambda_n >= 0 of
/// the periodic Laplacian -- strictly positive for ANY alpha > 0, so the
/// system is always symmetric positive-definite. There is no resonant
/// value of alpha to avoid here.
///////////////////////////////////////////////////////////////////////////

#ifndef GV_DOF_PERIOD
	#define GV_DOF_PERIOD 0b110   // periodic in x, y, and z
#endif

using Scalar_t     = double;
using Point_t      = gutil::Point<3,Scalar_t>;
using Box_t        = gutil::Box<3,Scalar_t>;

using Mesh_t       = GV::UnstructuredVoxelMesh<Scalar_t>;
using Elem_t       = typename Mesh_t::Elem_t;
using Vert_t       = typename Mesh_t::Vert_t;

using DOF_t        = GV::Keys::DOFS::VoxelQ1<GV_DOF_PERIOD>;
using DofHandler_t = GV::DofHandler<Mesh_t,DOF_t>;

using CoefHandler_t = GV::CoefHandler<DofHandler_t,Scalar_t,1>;

inline constexpr Scalar_t PI = 3.14159265358979323846;

// alpha > 0 -- the shift in -Delta(u) + alpha*u. Unlike genuine Helmholtz,
// ANY positive value is safe here; there is no resonance to avoid.
inline constexpr Scalar_t ALPHA = 5.0;

// Manufactured exact solution, evaluated pointwise -- used both to initialize
// nothing (the solve is what produces the coefficients) and to measure error
// against the true solution afterward.
inline constexpr Scalar_t exact_u(Scalar_t x, Scalar_t y, Scalar_t z) noexcept {
	return std::cos(PI*x) * std::cos(PI*y) * std::cos(PI*z);
}


///////////////////////////////////////////////////////////////////////////
/// Bilinear form kernel: -Delta(u) + alpha*u, i.e. H1(u,v) + alpha*L2(u,v)
/// Built directly via the kernel-composition operators (+ and *) rather
/// than a hand-written kernel, exercising the SumBilinearKernel /
/// ScaledBilinearKernel machinery. Note the PLUS sign on the L2 term here
/// -- this is what makes the assembled matrix positive-definite.
///////////////////////////////////////////////////////////////////////////
inline auto make_helmholtz_kernel() noexcept {
	return GV::H1BilinearKernel<true>{} + ALPHA*GV::L2BilinearKernel<true>{};
}
using HelmholtzKernel_t = decltype(make_helmholtz_kernel());
using BilinearForm_t    = GV::BilinearForm<4, Scalar_t, DofHandler_t, DofHandler_t, HelmholtzKernel_t>;


///////////////////////////////////////////////////////////////////////////
/// Use c[i]^2 int_D |grad(dof[i])|^2 as a per-dof error indicator
///////////////////////////////////////////////////////////////////////////
using ErrorIndicatorKernel_t = GV::H1BilinearKernel<true>;
using ErrorIndicatorForm_t   = GV::BilinearForm<2, Scalar_t, DofHandler_t, DofHandler_t, ErrorIndicatorKernel_t>;


///////////////////////////////////////////////////////////////////////////
/// Linear form kernel for the right-hand side: L(v) = int_D( f(x,y,z) * v ) dx
/// f is supplied via CRTP, matching the WeightedLinearKernel convention
/// established earlier -- eval_weight receives the physical (x,y,z)
/// coordinates at each quadrature point and must fill in f there.
///////////////////////////////////////////////////////////////////////////
struct HelmholtzForcingWeight {
	static constexpr bool NEEDS_GEO_POINTS  = true;
	static constexpr bool NEEDS_SCALAR_VALS = false;

	template<typename QuadRule_t>
	static GV::ScalarValueCache<QuadRule_t> build_weights(const GV::ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) noexcept {
		GV::ScalarValueCache<QuadRule_t> wt;
		
		auto x = qr.geo_x(), y=qr.geo_y(), z=qr.geo_z();
		GUTIL_SIMD()
		for (size_t i=0; i<wt.size(); ++i) {
			const Scalar_t f = std::cos(PI*x[i]) * std::cos(PI*y[i]) * std::cos(PI*z[i]);
			wt[i] = (3*PI*PI + ALPHA) * f;
		}
		return wt;
	}
};
using ForcingForm_t = GV::LinearForm<4, Scalar_t, DofHandler_t, GV::L2LinearKernel<true>, HelmholtzForcingWeight>;



void refine_solution(Mesh_t& mesh, DofHandler_t& d_handler, CoefHandler_t& c_handler) {
	GUTIL_ASSERT(mesh.is_current() && d_handler.is_current() && c_handler.is_current());
	
	//compute error indicator vector and threshold value for refinement
	ErrorIndicatorForm_t e_form(d_handler);
	std::vector<Scalar_t> errs(d_handler.n_dofs(), 0);
	e_form.construct_diagonal(errs);

	auto coefs = c_handler.get_coefs(0);
	Scalar_t min{0}, max{0}, mean{0};
	GUTIL_SIMD(reduction(min:min) reduction(max:max) reduction(+:mean))
	for (size_t i=0; i<errs.size(); ++i) {
		errs[i] *= coefs[i]*coefs[i];
		min = std::min(errs[i], min);
		max = std::max(errs[i], max);
		mean += errs[i];
	}
	mean /= static_cast<Scalar_t>(errs.size());

	GUTIL_LOG("min_err=", min, " max_err=", max, " mean_err=", mean);

	Scalar_t r_threshold = mean+0.5*(max-mean);
	
	//collect dofs for refine
	{
		auto lock = d_handler.begin_key_mask_unstable();
		for (size_t i=0; i<d_handler.n_dofs(); ++i) {
			if (errs[i]>r_threshold) {
					// std::cout << "refine dof? " << d_handler[i] << " mask: " << GV::print_bytes(d_handler.get_mask(d_handler[i])) << "\n";
				if (d_handler.can_refine(d_handler[i])) {
					d_handler.refine_quasi_hierarchical(d_handler[i]);
				}
			}
		}
		d_handler.end_key_mask_unstable();
	}

	//update mesh and coefficients
	d_handler.collect_dofs();
	mesh.process_refine();
	c_handler.prolong_coefs();
}

void update_solution(DofHandler_t& d_handler, CoefHandler_t& c_handler) {
	HelmholtzKernel_t kernel = make_helmholtz_kernel();
	BilinearForm_t b_form(d_handler, kernel);
	const size_t n = d_handler.n_dofs();
	GUTIL_LOG(n, " dofs");

	std::vector<Eigen::Triplet<Scalar_t>> triplets;
	{
		GUTIL_TIMER("assembling A (triplets)");
		b_form.build_triplets(triplets);
	}

	Eigen::SparseMatrix<Scalar_t> A(n, n);
	A.setFromTriplets(triplets.begin(), triplets.end());
	triplets.clear(); triplets.shrink_to_fit();

	// ---- Assemble the right-hand side b, via the linear form ----
	ForcingForm_t l_form(d_handler);
	Eigen::VectorXd b(n);
	b.setZero();
	{
		GUTIL_TIMER("assembling b (linear form)");
		l_form.evaluate_vector(GV::as_span(b));
	}

	// ---- Solve ----
	// link the unknowns in eigen to the coefficient handler
	Eigen::Map<Eigen::VectorXd> x(c_handler.get_coefs(0).data(), c_handler.get_coefs(0).size());
	{
		GUTIL_TIMER("solving");
		Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar_t>> solver;
		solver.compute(A);
		if (solver.info() != Eigen::Success) {
			GUTIL_LOG("ERROR: matrix factorization failed");
			return;
		}
		x = solver.solve(b);
	}
}

void compare_to_exact(const DofHandler_t& d_handler, const CoefHandler_t& c_handler) {
	const size_t n = d_handler.n_dofs();
	// ---- Compare against the manufactured exact solution ----
	// TODO: make linear forms to evaluate the integral L2 error
	Scalar_t max_err{0}, l2_err{0};
	for (size_t i=0; i<n; ++i) {
		DOF_t dof = d_handler[i];
		Point_t p = d_handler.mesh.geo_coord(Vert_t{dof.key});
		const Scalar_t u_exact = exact_u(p[0], p[1], p[2]);
		const Scalar_t err = std::abs(c_handler.get_coefs(0)[i] - u_exact);	//not accurate for non-uniform mesh
		max_err = std::max(max_err, err);
		l2_err += err*err;
	}
	l2_err = std::sqrt(l2_err / static_cast<Scalar_t>(n));

	GUTIL_LOG("max error = ", max_err, "   rms error = ", l2_err);
}



///////////////////////////////////////////////////////////////////////////
/// Assemble A (via triplets -> Eigen sparse), assemble b (via the linear
/// form), solve, then compare against the manufactured exact solution.
///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	size_t initial_depth = 2;
	size_t n_refines     = 3;
	if (argc > 1) { initial_depth = static_cast<size_t>(std::atoi(argv[1])); }
	if (argc > 2) { n_refines     = static_cast<size_t>(std::atoi(argv[2])); }

	Box_t domain{ {-1,-1,-1}, {1,1,1} };
	Mesh_t mesh(domain, initial_depth+n_refines);
	mesh.set_depth(initial_depth);

	DofHandler_t d_handler(mesh);
	d_handler.init_dofs();

	CoefHandler_t c_handler(d_handler);
	c_handler.init_coefs(0, [](DOF_t){return 0;});


	for (size_t i=0; i<n_refines; ++i) {
		update_solution(d_handler, c_handler);
		compare_to_exact(d_handler, c_handler);
		refine_solution(mesh, d_handler, c_handler);
	}

	


	{
		//save the solution
		GUTIL_TIMER("saving solution");
		mesh.collect_vertices();
		GUTIL_ASSERT(mesh.is_current());
		GUTIL_ASSERT(d_handler.is_current());

		std::vector<Scalar_t> scalar_vals(mesh.n_vertices());
		scalar_vals = c_handler.evaluate(0, mesh.vertex_begin(), mesh.vertex_end());

		mesh.save_as_binary("screened_poisson.vtk");

		auto pt_field_lookup = GV::make_index_lookup<float>(
			[&](uint64_t idx){ return (float) scalar_vals[idx]; }, "scalar_field");
		auto pt_coef_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {
				auto d_vtx = d_handler.get_dof_vertex(vtx);
				return d_vtx.exists() && d_handler.is_active_stable(DOF_t{d_vtx}) ? 
							(float)c_handler.coefs[0][d_handler.global_number(DOF_t{d_vtx})] : -1;
			}, "scalar_coef");
		auto pt_dof_key_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {
				auto d_vtx = d_handler.get_dof_vertex(vtx);
				return d_vtx.exists() && d_handler.is_active_stable(DOF_t{d_vtx}) ?
							std::array<int32_t,4>{(int32_t)d_vtx.depth(), (int32_t)d_vtx.i(), (int32_t)d_vtx.j(), (int32_t)d_vtx.k()} :
							std::array<int32_t,4>{-1,-1,-1,-1};
			}, "dof_key");

		mesh.append_point_data_field_binary("screened_poisson.vtk", "point", pt_field_lookup, pt_coef_lookup, pt_dof_key_lookup);
	}



	return 0;
}