#include "gutil.hpp"

#include "simd_keys/dofs/voxelQ1/voxel_Q1_interface.hpp"
#include "mesh/voxel_mesh_unstructured(NEW).hpp"
#include "diffuse_domain/signed_distance.hpp"
#include "fem/handlers/dofhandler.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"
#include "fem/handlers/coef_handler.hpp"

/////////////////////////////////////////////////////////////////
/// Compile-time options
///
///   GV_TEST_DOMAIN_PERIOD  0bzyx periodicity of the MESH (e.g. 6=0b110 -> periodic in y,z)
///   GV_TEST_DOF_PERIOD     0bzyx periodicity of the DOFS (independent of the mesh's own)
///   GV_TEST_MAX_DEPTH      max voxel mesh depth (>10 not recommended, >15 unsupported)
///   GV_TEST_DOMAIN_SIZE    domain half-sidelength per axis, centered at the origin
/////////////////////////////////////////////////////////////////
#ifndef GV_TEST_DOMAIN_PERIOD
	#define GV_TEST_DOMAIN_PERIOD 7
#endif
#ifndef GV_TEST_DOF_PERIOD
	#define GV_TEST_DOF_PERIOD 0
#endif
#ifndef GV_TEST_DOMAIN_SIZE
	#define GV_TEST_DOMAIN_SIZE 1
#endif

/////////////////////////////////////////////////////////////////
/// Aliases
/////////////////////////////////////////////////////////////////
using Point_t       = gutil::Point<3,double>;
using Box_t         = gutil::Box<3,double>;
using Sphere_t      = gutil::Sphere<3,double>;

using Mesh_t        = GV::UnstructuredVoxelMesh<10>;
using MeshVert_t    = typename Mesh_t::VoxelVertex;
using MeshElem_t    = typename Mesh_t::VoxelElement;
using Assembly_t    = GV::SignedDistanceSpheres<double,GV_TEST_DOMAIN_PERIOD>;

using DOF_t         = GV::Keys::DOFS::VoxelQ1<GV_TEST_DOF_PERIOD>;
using Handler_t     = GV::DofHandler<Mesh_t,DOF_t>;
using DofVert_t     = typename Handler_t::DofVert_t;
using DofElem_t     = typename Handler_t::DofElem_t;

using CoefHandler_t = GV::CoefHandler<Handler_t,double,1>;

inline constexpr double DOMAIN_SIZE = GV_TEST_DOMAIN_SIZE;
inline constexpr size_t MAX_DEPTH   = Mesh_t::MAX_DEPTH;
inline constexpr Box_t domain{ {-DOMAIN_SIZE,-DOMAIN_SIZE,-DOMAIN_SIZE},
							   { DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_SIZE} };

/////////////////////////////////////////////////////////////////
/// Runtime test configuration (from argv)
/////////////////////////////////////////////////////////////////
struct TestConfig {
	std::string test_name    = "dofhandler";
	size_t      n_spheres    = 100;
	size_t      seed         = 0;
	double      min_r        = 0.1;
	double      max_r        = 0.5;
	size_t      initial_depth= MAX_DEPTH/3;
	double      refine_tol   = 0.1;   	// signed-distance tolerance for boundary refinement
	double      unrefine_x   = 0.5;  	// normalized-x threshold for the unrefinement demo region
	uint8_t		max_depth    = 6;		// max depth of the mesh
	uint8_t     n_refine     = 1;       // target number of refinements
};

TestConfig parse_args(int argc, char* argv[]) {
	TestConfig cfg;
	std::vector<std::string> args(argv, argv+argc);
	for (size_t i=0; i<args.size(); ++i) {
		if      (args[i] == "-N")    { cfg.n_spheres     = atoi(args[++i].c_str()); }
		else if (args[i] == "-R0")   { cfg.min_r         = atof(args[++i].c_str()); }
		else if (args[i] == "-R1")   { cfg.max_r         = atof(args[++i].c_str()); }
		else if (args[i] == "-S")    { cfg.seed          = atoi(args[++i].c_str()); }
		else if (args[i] == "-name") { cfg.test_name     = args[++i]; }
		else if (args[i] == "-ID")   { cfg.initial_depth = atoi(args[++i].c_str()); }
		else if (args[i] == "-TOL")  { cfg.refine_tol    = atof(args[++i].c_str()); }
		else if (args[i] == "-UX")   { cfg.unrefine_x    = atof(args[++i].c_str()); }
		else if (args[i] == "-DEPTH"){ cfg.max_depth     = atoi(args[++i].c_str()); }
		else if (args[i] == "-NR")   { cfg.n_refine      = atoi(args[++i].c_str()); }
	}
	return cfg;
}

/////////////////////////////////////////////////////////////////
/// Generate a pseudo-random periodic collection of non-overlapping spheres.
/////////////////////////////////////////////////////////////////
Assembly_t generate_assembly(const TestConfig& cfg) {
	GUTIL_TIMER("Generating Periodic Assembly: START");
	Assembly_t assembly{domain};

	auto random_point = gutil::UniformRandomPoint<Point_t,true>();
	random_point.set_parameters(-DOMAIN_SIZE, DOMAIN_SIZE);
	random_point.set_seed(cfg.seed);

	auto random_radius = gutil::UniformRandomPoint<Point_t,true>();
	random_radius.set_parameters(cfg.min_r, cfg.max_r);
	random_radius.set_seed(cfg.seed);

	const size_t max_attempts = 500 * cfg.n_spheres;
	size_t attempts = 0;
	while (assembly.size() < cfg.n_spheres && attempts < max_attempts) {
		++attempts;
		Sphere_t candidate(random_point(), random_radius.scalar());
		if (!assembly.collides(candidate)) { assembly.push_back(std::move(candidate)); }
	}

	if (assembly.size() < cfg.n_spheres) {
		GUTIL_ERROR("only placed ", assembly.size(), "/", cfg.n_spheres,
							" disjoint periodic spheres after ", max_attempts, " attempts.\n",
							"Try a smaller number of spheres or radii.");
	}

	gutil::write_spheres_to_file(cfg.test_name + "_spheres.txt", assembly.as_cspan());
	return assembly;
}

/////////////////////////////////////////////////////////////////
/// Phase: mesh + dof handler setup, initial (uniform) coefficient field
/////////////////////////////////////////////////////////////////
void setup_mesh_and_dofs(Mesh_t& mesh, Handler_t& handler, CoefHandler_t& coef_handler) {
	GUTIL_TIMER("initializing mesh");

	mesh.update_unstructured();
	handler.init_dofs();
	handler.collect_dofs();

	GUTIL_LOG("dofhandler has ", handler.n_dofs(), " dofs");
	GUTIL_LOG("mesh has ", mesh.n_elements(), " elements");

	coef_handler.init_coefs(0, [&mesh](DOF_t dof){
		auto pt = mesh.geo_coord(static_cast<MeshVert_t>(dof));
		(void)pt;
		return 1.0;
	});
}

/////////////////////////////////////////////////////////////////
/// Collect the active mesh elements whose geometric distance to the
/// assembly boundary is within tol. This is the "which elements does
/// the batch refine/unrefine act on" step -- everything downstream of
/// this is just handler.refine_quasi_hierarchical(span)/unrefine_quasi_hierarchical(span).
/////////////////////////////////////////////////////////////////
std::vector<MeshElem_t> collect_near_boundary(const Mesh_t& mesh, const Assembly_t& assembly, double tol) {
	std::vector<MeshElem_t> elems;
	for (auto it=mesh.element_begin(); it!=mesh.element_end(); ++it) {
		if (gutil::norm2(it->normalized_center()) < 0.5 ) {
		// if (it->normalized_center()[0] < 0.5 || it->normalized_center()[1]<0.5) {
		// if (std::abs(assembly.signed_distance(mesh.geo_center(*it))) < tol) {
			elems.push_back(*it);
		}
	}
	return elems;
}

/////////////////////////////////////////////////////////////////
/// Phase: refine near the sphere-assembly boundary, one depth level at a time,
/// using the batch handler.refine_quasi_hierarchical(span<const MeshElem_t>) directly.
///
/// update_coefs() is called once per level -- calling it after more than one
/// level of refinement has been applied will silently drop contributions
/// from dofs that were refined past their immediate children.
/////////////////////////////////////////////////////////////////
void refine_near_boundary(Mesh_t& mesh, Handler_t& handler, CoefHandler_t& coef_handler,
						   const Assembly_t& assembly, const TestConfig& cfg) {
	GUTIL_TIMER("refining dofs");

	uint8_t depth = cfg.initial_depth;
	for (uint8_t n=0; n<cfg.n_refine && depth<mesh.max_depth; ++n, ++depth) {

		std::vector<MeshElem_t> near_boundary = collect_near_boundary(mesh, assembly, cfg.refine_tol);
		GUTIL_LOG("depth ", depth, ": refinement  ", near_boundary.size(), " elements");

		size_t n_refined = handler.refine_quasi_hierarchical(near_boundary);

		// size_t n_els = mesh.n_elements();
		// size_t n_refined = handler.refine_quasi_hierarchical(mesh.element_begin(), mesh.element_begin()+n_els/2);
		GUTIL_LOG("  -> ", n_refined, " dofs refined");

		mesh.process_refine();
		mesh.update_unstructured();
		coef_handler.update_coefs();
	}

	// final pass: also grow the mesh itself slightly ahead of the boundary,
	// so the last refined dof level has proper 2-1 support
	mesh.process_refine([&](MeshElem_t el) {
		return assembly.signed_distance(mesh.geo_center(el)) < 2*cfg.refine_tol;
	});
	mesh.update_unstructured();
	coef_handler.update_coefs();

	for (DOF_t dof : handler.active_dofs) {
		if (!mesh.is_geometrically_conformal(DofVert_t{dof.key})) {
			GUTIL_ERROR("\tERROR: ", dof, " is not conformal");
		}
	}

	GUTIL_LOG("dofhandler has ", handler.n_dofs(), " dofs");
	GUTIL_LOG("mesh has ", mesh.n_elements(), " elements");
}

/////////////////////////////////////////////////////////////////
/// Phase: unrefine a demo region (normalized x > threshold), coalescing
/// each qualifying dof's children back into its own, coarser support --
/// using the batch handler.unrefine_quasi_hierarchical(span<const MeshElem_t>) directly.
/////////////////////////////////////////////////////////////////
void unrefine_demo_region(Mesh_t& mesh, Handler_t& handler, CoefHandler_t& coef_handler,
						   const TestConfig& cfg) {
	GUTIL_TIMER("unrefining dofs");

	std::vector<MeshElem_t> region;
	for (auto it=mesh.element_begin(); it!=mesh.element_end(); ++it) {
		GUTIL_ASSERT(mesh.is_active(*it));
		if (it->normalized_center()[0] > cfg.unrefine_x) {
			region.push_back(*it);
		}
	}
	GUTIL_LOG("unrefining ", region.size(), " elements in the demo region");

	size_t n_unrefined = handler.unrefine_quasi_hierarchical(std::span<const MeshElem_t>{region});
	GUTIL_LOG("  -> ", n_unrefined, " dofs actually unrefined");

	handler.collect_dofs();
	coef_handler.update_coefs();

	for (DOF_t dof : handler.active_dofs) {
		if (!mesh.is_geometrically_conformal(DofVert_t{dof.key})) {
			GUTIL_ERROR(dof, " is not conformal");
		}
	}

	GUTIL_LOG("dofhandler has ", handler.n_dofs(), " dofs");
	GUTIL_LOG("mesh has ", mesh.n_elements(), " elements");
}

/////////////////////////////////////////////////////////////////
/// Phase: drop mesh elements that no longer have active dof.
/////////////////////////////////////////////////////////////////
void remove_unsupported_elements(Mesh_t& mesh, Handler_t& handler) {
	GUTIL_TIMER("remove mesh elements with no dofs");

	for (auto it=mesh.element_begin(); it!=mesh.element_end(); ++it) {
		std::vector<DOF_t> dofs;
		std::vector<size_t> numbers;
		handler.get_active_dofs_quasi_hierarchical(*it, dofs, numbers);
		if (dofs.empty()) { mesh.unrefine(*it); }
	}

	handler.collect_dofs();
	mesh.process_unrefine();
	mesh.update_unstructured();
	for (DOF_t dof : handler.active_dofs) {
		if (!mesh.is_geometrically_conformal(DofVert_t{dof.key})) {
			GUTIL_ERROR(dof, " is not conformal");
		}
	}

	GUTIL_LOG("dofhandler has ", handler.n_dofs(), " dofs");
	GUTIL_LOG("mesh has ", mesh.n_elements(), " elements");
}

/////////////////////////////////////////////////////////////////
/// Phase: evaluate the field at every mesh vertex and save results to disk.
/////////////////////////////////////////////////////////////////
void evaluate_and_save(Mesh_t& mesh, Handler_t& handler, CoefHandler_t& coef_handler,
					   const Assembly_t& assembly, const TestConfig& cfg) {
	mesh.collect_vertices();
	handler.collect_dofs();

	std::vector<double> scalar_vals(mesh.n_vertices());
	{
		GUTIL_LOG("Final mesh size: ", mesh.n_elements(), " elements and ", mesh.n_vertices(), " vertices");
		GUTIL_LOG("Final number of dofs: ", handler.n_dofs());
		GUTIL_TIMER("evaluating field at mesh vertices");

		scalar_vals = coef_handler.evaluate(0, mesh.vertex_begin(), mesh.vertex_end());
	}

	GUTIL_LOG("saving file");
	const std::string filename = cfg.test_name + "_dof.vtk";
	mesh.save_as_binary(filename);

	auto pt_field_lookup = GV::make_index_lookup<double>(
			[&](uint64_t idx){ return scalar_vals[idx]; }, "scalar_field");

	auto pt_coef_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) {
				auto d_vtx = handler.get_dof_vertex(vtx);
				return d_vtx.exists() && handler.is_active(DOF_t{d_vtx}) ? (int64_t)coef_handler.coefs[0][handler.global_number(DOF_t{d_vtx})] : -1;
			}, "scalar_coef");

	auto pt_sd_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) { return assembly.signed_distance(mesh.geo_coord(vtx)); },
			"signed_distance");

	auto pt_dof_active_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) {
				auto d_vtx = handler.get_dof_vertex(vtx);
				return d_vtx.exists() && handler.is_active(DOF_t{d_vtx}) ? (int64_t)handler.global_number(DOF_t{d_vtx}) : -1;
			}, "active_dof_index");

	auto pt_dof_refinable_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) {
				auto d_vtx = handler.get_dof_vertex(vtx);
				int val = -1;
				if (d_vtx.exists() && handler.is_active(DOF_t{d_vtx}) ) {
					val = 10*(int)handler.is_refinable(DOF_t{d_vtx}) + (int)handler.is_unrefinable(DOF_t{d_vtx});
				}
				return val;
			}, "dof_ref_unref");

	auto pt_dof_key_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) {
				auto d_vtx = handler.get_dof_vertex(vtx);
				return d_vtx.exists() && handler.is_active(DOF_t{d_vtx}) ?
							std::array<int64_t,4>{(int64_t)d_vtx.depth(), (int64_t)d_vtx.i(), (int64_t)d_vtx.j(), (int64_t)d_vtx.k()} :
							std::array<int64_t,4>{-1,-1,-1,-1};
			}, "dof_key");

	auto pt_mesh_key_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) {
				return vtx.exists() ?
							std::array<int64_t,4>{(int64_t)vtx.depth(), (int64_t)vtx.i(), (int64_t)vtx.j(), (int64_t)vtx.k()} :
							std::array<int64_t,4>{-1,-1,-1,-1};
			}, "mesh_vtx_key");

	auto pt_vtx_key_lookup = GV::make_feature_lookup<MeshVert_t>(
		[&](MeshVert_t vtx) {
			return std::array<int64_t,4>{(int64_t)vtx.depth(), (int64_t)vtx.i(), (int64_t)vtx.j(), (int64_t)vtx.k()};
		}, "viewed_vtx_key");

	auto el_dof_count_lookup = GV::make_feature_lookup<MeshElem_t>(
			[&](MeshElem_t el) {
				std::vector<DOF_t> dofs;
				std::vector<size_t> idxs;
				handler.get_active_dofs_full_hierarchical(el, dofs, idxs);
				return dofs.size();
			}, "n_active_dofs");

	auto el_idx_lookup = GV::make_feature_lookup<MeshElem_t>(
			[&](MeshElem_t el) {
				return el.exists() ?
						std::array<int64_t,4>{(int64_t)el.i(), (int64_t)el.j(), (int64_t)el.k(), (int64_t)el.depth_linear_index()} :
						std::array<int64_t,4>{-1,-1,-1,-1};

			}, "el_ijk_morton");

	auto el_depth_lookup = GV::make_feature_lookup<MeshElem_t>(
			[](MeshElem_t el) { return el.depth(); }, "depth");

	mesh.append_point_data_field_binary(filename, "point",
			pt_sd_lookup, pt_dof_active_lookup, pt_dof_refinable_lookup,
			pt_dof_key_lookup, pt_field_lookup, pt_coef_lookup, pt_vtx_key_lookup);
	mesh.append_cell_data_field_binary(filename, "element",
			el_dof_count_lookup, el_idx_lookup, el_depth_lookup);
}

/////////////////////////////////////////////////////////////////
/// Top-level test driver
/////////////////////////////////////////////////////////////////
void test_dof_handler(const TestConfig& cfg, const Assembly_t& assembly) {
	GUTIL_LOG("test_dof_handler");

	Mesh_t mesh(domain, cfg.initial_depth);
	Handler_t handler(mesh);
	CoefHandler_t coef_handler(handler);

	setup_mesh_and_dofs(mesh, handler, coef_handler);

	refine_near_boundary(mesh, handler, coef_handler, assembly, cfg);

	// unrefine_demo_region(mesh, handler, coef_handler, cfg);
	// remove_unsupported_elements(mesh, handler);

	evaluate_and_save(mesh, handler, coef_handler, assembly, cfg);
}

int main(int argc, char* argv[]) {
	TestConfig cfg = parse_args(argc, argv);
	Assembly_t assembly = generate_assembly(cfg);
	test_dof_handler(cfg, assembly);
	return 0;
}