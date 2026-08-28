#include "gutil.hpp"
#include "geovox.hpp"

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <cmath>
#include <iostream>
#include <string>


/////////////////////////////////////////////////////////////
/// See "Diffuse Domain Methods with Dirichlet Boundary Conditions"
/// by Luke Benfield and Andreas Dedner.
/// We extend the problem -laplace(u)=f on a ball of radius R (Omega)
/// with Dirichlet data g to the diffuse domain problem
/// -div(phi*grad(u)) + (1-phi)(u-g)/eps^3 = phi*f
/// with f and g modified near the boundary of the original domain.
/// This gives a weak form: find u s.t. a(u,v)=l(v) for all v where
///		a(u,v) = int_B(grad(u)*grad(v) *phi dx) + int_B(u*v * (1-phi)/eps^3 dx)
/// 	l(v)   = int_B(f*v *phi dx) + int_B(g*v *(1-phi)/eps^3 dx)
/// Where B is a superset of Omega and
/// phi(x) = 0.5*(1 - tanh(3r(x)/eps)) with r(x) the signed distance from x
/// in B to the boundary of Omega.
/////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////
/// A few compile-time settings
/////////////////////////////////////////////////////////////
inline constexpr int N_QUAD_POINTS_PER_AXIS = 3;
inline constexpr uint8_t PERIODIC_BC = 0b111;

/////////////////////////////////////////////////////////////
/// A few aliases
/////////////////////////////////////////////////////////////
using Scalar_t       = float;
using Point_t        = gutil::Point<3,Scalar_t>;
using Box_t          = gutil::Box<3,Scalar_t>;
using Sphere_t       = gutil::Sphere<3,Scalar_t>;

using Assembly_t     = GV::SignedDistanceSpheres<Scalar_t,PERIODIC_BC,true,1>;
using MeshHandler_t  = GV::DiffuseDomainMeshHandler<Assembly_t>;
using Mesh_t         = typename MeshHandler_t::Mesh_t;
using Elem_t         = typename MeshHandler_t::Mesh_t::Elem_t;
using Vert_t         = typename MeshHandler_t::Mesh_t::Vert_t;

using DOF_t          = GV::Keys::DOFS::VoxelQ1<PERIODIC_BC>;
using DofHandler_t   = GV::DofHandler<Mesh_t,DOF_t>;
using CoefHandler_t  = GV::CoefHandler<DofHandler_t,Scalar_t,1>;

using InteriorWeight_t = GV::AssemblyPhaseFieldWeight<true,  Assembly_t>;
using ExteriorWeight_t = GV::AssemblyPhaseFieldWeight<false, Assembly_t>;


/////////////////////////////////////////////////////////////
/// A few run-time settings
///
///	On the refinement method: encoded bit-wise 0bCBA
///	A=1 -> Geometric refinement	 
///
/////////////////////////////////////////////////////////////
struct TestConfig {
	std::string test_name     = "";
	size_t      initial_depth = 3;		// depth of the initial mesh
	Scalar_t    eps_scale     = 1;		// eps = eps_scale*min_h
	int    		n_refine      = 3;		// number of refinements
	Scalar_t	g_tol         = 3;   	// distance (as a multiple of h, to element center) to refine near the domain boundary
	int  		ref_method    = 0b1;	// refinement technique(s)

	std::string to_string() const {
		std::string str = "";
		str += "test_name: " + test_name + "\n";
		str += "initial_depth: " + std::to_string(initial_depth) + "\n";
		str += "eps_scale: " + std::to_string(eps_scale) + "\n";
		str += "n_refine: " + std::to_string(n_refine) + "\n";
		str += "g_tol: " + std::to_string(g_tol) + "\n";
		str += "ref_method: " + std::to_string(ref_method) + "\n";
		return str;
	}
};

TestConfig parse_args(int argc, char* argv[]) {
	TestConfig cfg;
	std::vector<std::string> args(argv, argv+argc);
	for (size_t i=0; i<args.size(); ++i) {
		if      (args[i] == "-NAME") 	{ cfg.test_name     = args[++i]; }
		else if (args[i] == "-ID")   	{ cfg.initial_depth = atoi(args[++i].c_str()); }
		else if (args[i] == "-NR")   	{ cfg.n_refine      = atoi(args[++i].c_str()); }
		else if (args[i] == "-GTOL") 	{ cfg.g_tol         = atof(args[++i].c_str()); }
		else if (args[i] == "-RMETHOD") { cfg.ref_method    = atoi(args[++i].c_str()); }
	}
	return cfg;
}


/////////////////////////////////////////////////////////////
/// A few global variables
/////////////////////////////////////////////////////////////
inline TestConfig			 cfg{};			//track the run-time options
inline std::vector<Scalar_t> mesh_size{};	//track minimum mesh size
inline std::vector<size_t>   n_dofs{};		//track the number of degrees of freedom/basis functions
inline std::vector<Scalar_t> l2_err{};		//track the L2 error


/////////////////////////////////////////////////////////////
/// Set up the exact solution and a proxy-linear form to 
/// integrate the error.
/////////////////////////////////////////////////////////////
inline constexpr Scalar_t RADIUS = 1;
inline constexpr Point_t  CENTER(0,0,0);
inline constexpr Box_t    DOMAIN(-1.5*Point_t::Filled(RADIUS), 1.5*Point_t::Filled(RADIUS));

inline constexpr auto u_exact = [](Scalar_t x, Scalar_t y, Scalar_t z) constexpr noexcept {
	return RADIUS*RADIUS - (x*x+y*y+z*z);
};
inline constexpr auto f_source   = [](Scalar_t, Scalar_t, Scalar_t) constexpr noexcept { return Scalar_t{6}; };
inline constexpr auto g_boundary = [](Scalar_t, Scalar_t, Scalar_t) constexpr noexcept { return Scalar_t{0}; };

using L2ErrorWeight_t = GV::L2DiffuseErrorWeight<Assembly_t, u_exact>;
using ErrorForm_t     = GV::LinearForm<N_QUAD_POINTS_PER_AXIS+1, Scalar_t, DofHandler_t, GV::L2LinearKernel<>, L2ErrorWeight_t>;


/////////////////////////////////////////////////////////////
/// Set up a string to record compile time settings
/////////////////////////////////////////////////////////////
std::string compile_opts() {
	std::string str = "";
	str += "N_QUAD_POINTS_PER_AXIS= " + std::to_string(N_QUAD_POINTS_PER_AXIS) + "\n";
	str += "PERIODIC_BC= " + std::to_string(PERIODIC_BC) + "\n";
	str += "RADIUS= " + std::to_string(RADIUS) + "\n";
	str += "CENTER= " + gutil::to_string(CENTER) + "\n";
	str += "DOMAIN= " + gutil::to_string(DOMAIN) + "\n";
	return str;
}


/////////////////////////////////////////////////////////////
/// Refine by geometry
/////////////////////////////////////////////////////////////
void geometry_refine(MeshHandler_t& m_handler, DofHandler_t& d_handler, CoefHandler_t& c_handler) {
	//select elements to refine
	Scalar_t tol = cfg.g_tol * m_handler.min_element_size();
	std::vector<Elem_t> elems = m_handler.mesh.select_elements([&m_handler, tol](Elem_t el){
		const Point_t pt = m_handler.mesh.geo_center(el);
		return gutil::abs(m_handler.assembly.signed_distance(pt)) < tol;
	});

	//refine the dofs on the elements
	d_handler.refine_quasi_hierarchical(elems);

	//refine the mesh elements so the new basis functions can be resolved
	m_handler.mesh.process_refine();

	//update the coefficients
	c_handler.prolong_coefs();
}


/////////////////////////////////////////////////////////////
/// Solve the diffuse problem on the current mesh
/////////////////////////////////////////////////////////////
void solve_with_spmat(MeshHandler_t& m_handler, DofHandler_t& d_handler, CoefHandler_t& c_handler) {
	using EigenVec       = Eigen::Matrix<Scalar_t, Eigen::Dynamic, 1>;
	using EigenSpMat	 = Eigen::SparseMatrix<Scalar_t, Eigen::ColMajor>;
	using Triplet    	 = GV::Triplet<Scalar_t,EigenSpMat::StorageIndex,Eigen::ColMajor>;
	
	//a custom diagonal pre-conditioner seems to be faster than Eigen's (better parallelism is likely)
	using Preconditioner = GV::DiagonalPreconditioner<std::vector<Scalar_t>>;
	using EigenSolver    = Eigen::ConjugateGradient<EigenSpMat, Eigen::Lower|Eigen::Upper, Preconditioner>;

	GUTIL_TIMER("Solving system by building the sparse matrix");
	std::cout << m_handler.mesh << "\n" << d_handler << "\n";

	const size_t n = d_handler.n_dofs();
	std::vector<Triplet> triplets;
	EigenSpMat 			 A(n, n);
	EigenVec   			 b = EigenVec::Zero(n);
	Eigen::Map<EigenVec> x(c_handler.get_coefs(0).data(), c_handler.get_coefs(0).size());

	//update epsilon (a static variable) in the diffuse domain weights
	const Scalar_t eps = cfg.eps_scale * m_handler.min_element_size();
	GV::AssemblyDiffuseDomain<Assembly_t>::SetAssembly(m_handler.assembly);
	GV::AssemblyDiffuseDomain<Assembly_t>::SetEps(eps);

	InteriorWeight_t interior_wt{};	//phi
	auto penalty_wt = (Scalar_t{1}/(eps*eps*eps)) * ExteriorWeight_t{}; //(1-phi)/eps^3

	{
		GUTIL_TIMER("Building triplets");
		auto poisson_form = GV::MakeBilinearForm<N_QUAD_POINTS_PER_AXIS,Scalar_t>(d_handler, GV::H1BilinearKernel_SW{}, interior_wt);

		auto penalty_form = GV::MakeBilinearForm<N_QUAD_POINTS_PER_AXIS,Scalar_t>(d_handler, GV::L2BilinearKernel_SW{}, penalty_wt);

		//each build_triplets de-duplicates it's own list
		//let eigen sum the entries from both parts of the form
		poisson_form.build_triplets(triplets);
		penalty_form.build_triplets(triplets);

	}
	{
		GUTIL_TIMER("Building SpMatrix");
		A.setFromTriplets(triplets.begin(), triplets.end());
		triplets.clear();
		triplets.shrink_to_fit();
	}
	{
		GUTIL_TIMER("Building RHS");
		GV::ModifiedFunctionKernelWeight<Assembly_t, f_source> source_wt{};
		GV::ModifiedFunctionKernelWeight<Assembly_t, g_boundary> boundary_wt{};

		auto source_form = GV::MakeLinearForm<N_QUAD_POINTS_PER_AXIS,Scalar_t>(d_handler, GV::L2LinearKernel_W{}, interior_wt*source_wt);
		auto boundary_form = GV::MakeLinearForm<N_QUAD_POINTS_PER_AXIS,Scalar_t>(d_handler, GV::L2LinearKernel_W{}, penalty_wt*boundary_wt);
		
		//each call to evaluate_vector accumulates into the provided span
		source_form.evaluate_vector(GV::as_span(b));
		boundary_form.evaluate_vector(GV::as_span(b));
	}
	{
		GUTIL_TIMER("Solving system");
		EigenSolver solver;
		solver.compute(A);
		x = solver.solveWithGuess(b, x);

		GUTIL_LOG("CG iterations: ", solver.iterations());
		if (solver.info() != Eigen::Success) {
			GUTIL_LOG("WARNING: CG did not converge");
		}
	}
}


/////////////////////////////////////////////////////////////
/// Log the solution
/////////////////////////////////////////////////////////////
void log_solution(const MeshHandler_t& m_handler, const DofHandler_t& d_handler, const CoefHandler_t& c_handler) {
	//compute the L2 error
	ErrorForm_t e_form(d_handler);
	auto err = gutil::sqrt(e_form.integrate_weight(c_handler.get_coefs(0)));

	GUTIL_LOG("L2 error: ", err, " n_dofs= ", d_handler.n_dofs(), " min_h= ", m_handler.min_element_size());
	mesh_size.push_back(m_handler.min_element_size());
	n_dofs.push_back(d_handler.n_dofs());
	l2_err.push_back(err);
}


/////////////////////////////////////////////////////////////
/// Initialize the problem
/////////////////////////////////////////////////////////////
void init(MeshHandler_t& m_handler, DofHandler_t& d_handler, CoefHandler_t& c_handler) {
	//add the sphere to the assembly
	m_handler.assembly.push_back(Sphere_t{CENTER,RADIUS});

	//set the mesh to a uniform depth
	m_handler.mesh.set_depth(cfg.initial_depth);

	//add dofs at the conformal mesh locations
	d_handler.init_dofs();

	//assign initial coefficients (all zeros)
	c_handler.init_coefs();
}


/////////////////////////////////////////////////////////////
/// Save the solution
/////////////////////////////////////////////////////////////
void save(int number, MeshHandler_t& m_handler, const DofHandler_t& d_handler, const CoefHandler_t& c_handler) {
	if (cfg.test_name.empty()) {return;}

	std::string filename = cfg.test_name + "_" + std::to_string(number) + ".vtk";
	GUTIL_TIMER("Saving solution as ", filename);

	m_handler.mesh.collect_vertices();
	const Assembly_t& assembly = m_handler.assembly;
	const Mesh_t& 	  mesh = m_handler.mesh;

	mesh.save_as_binary(filename);
	Scalar_t eps = GV::AssemblyDiffuseDomain<Assembly_t>::eps;

	auto sd_lookup = GV::make_feature_lookup<Vert_t>(
		[&](Vert_t vtx) { return assembly.signed_distance(mesh.geo_coord(vtx)); }, "signed_distance");

	auto phi_lookup = GV::make_feature_lookup<Vert_t>(
		[&](Vert_t vtx) { return assembly.heaviside(mesh.geo_coord(vtx), eps); }, "phi");

	std::vector<Scalar_t> u_vals = c_handler.evaluate(0, mesh.vertex_begin(), mesh.vertex_end());
	auto u_lookup = GV::make_index_lookup<Scalar_t>(
		[&](uint64_t idx) { return u_vals[idx]; }, "u");

	mesh.append_point_data_field_binary(filename, "point_data", sd_lookup, phi_lookup, u_lookup);
}


/////////////////////////////////////////////////////////////
/// Main program
/////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	GUTIL_TIMER("Convergence test for diffuse poisson on a sphere");

	/// Initialize
	cfg = parse_args(argc, argv);
	MeshHandler_t m_handler(DOMAIN, cfg.initial_depth+cfg.n_refine);
	DofHandler_t  d_handler(m_handler.mesh);
	CoefHandler_t c_handler(d_handler);
	init(m_handler, d_handler, c_handler);

	/// Initial Solve
	solve_with_spmat(m_handler, d_handler, c_handler);
	save(0, m_handler, d_handler, c_handler);
	log_solution(m_handler, d_handler, c_handler);

	/// Refine And Solve
	for (int i=0; i<cfg.n_refine; ++i) {
		GUTIL_LOG("Refinement ", i+1, "/", cfg.n_refine);
		geometry_refine(m_handler, d_handler, c_handler);
		solve_with_spmat(m_handler, d_handler, c_handler);
		save(i+1, m_handler, d_handler, c_handler);
		log_solution(m_handler, d_handler, c_handler);
	}

	/// Print Summary
	std::cout << "\nSettings Summary\n" << compile_opts() << cfg.to_string() << "\n";

	std::cout << "\nResult Summary:\n";
	std::cout << "n_dofs:"; for (auto n : n_dofs) {std::cout << "\t" << n;}
	std::cout << "\n";
	std::cout << "mesh_size:"; for (auto n : mesh_size) {std::cout << "\t" << n;}
	std::cout << "\n";
	std::cout << "l2_err:"; for (auto n : l2_err) {std::cout << "\t" << n;}
	std::cout << "\n";
}

