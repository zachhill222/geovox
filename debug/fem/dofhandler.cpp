#include "gutil.hpp"
#include "geovox.hpp"

/////////////////////////////////////////////////////////////////
/// Compile-time options
///
///   GV_TEST_DOMAIN_PERIOD  0bzyx periodicity of the MESH (e.g. 6=0b110 -> periodic in y,z)
/////////////////////////////////////////////////////////////////
#ifndef GV_TEST_DOF_PERIOD
	#define GV_TEST_DOF_PERIOD 1
#endif

/////////////////////////////////////////////////////////////////
/// Aliases
/////////////////////////////////////////////////////////////////
using Scalar_t 		= double;
using Point_t       = gutil::Point<3,Scalar_t>;
using Box_t         = gutil::Box<3,Scalar_t>;
using Sphere_t      = gutil::Sphere<3,Scalar_t>;

using Mesh_t        = GV::UnstructuredVoxelMesh<Scalar_t>;
using MeshVert_t    = typename Mesh_t::Vert_t;
using MeshElem_t    = typename Mesh_t::Elem_t;

using DOF_t         = GV::Keys::DOFS::VoxelQ1<GV_TEST_DOF_PERIOD>;
using Handler_t     = GV::DofHandler<Mesh_t,DOF_t>;
using DofVert_t     = typename Handler_t::DofVert_t;
using DofElem_t     = typename Handler_t::DofElem_t;

using CoefHandler_t = GV::CoefHandler<Handler_t,Scalar_t,1>;

inline constexpr Box_t domain{ {-1,-1,-1},
							   { 1, 1, 1} };

/////////////////////////////////////////////////////////////////
/// Runtime test configuration (from argv)
/////////////////////////////////////////////////////////////////
struct TestConfig {
	std::string test_name    = "dofhandler";
	size_t      initial_depth= 0;
	uint8_t     n_refine     = 1;
	uint8_t		n_unrefine   = 1;
};

TestConfig parse_args(int argc, char* argv[]) {
	TestConfig cfg;
	std::vector<std::string> args(argv, argv+argc);
	for (size_t i=0; i<args.size(); ++i) {
		if 		(args[i] == "-name") { cfg.test_name     = args[++i]; 				}
		else if (args[i] == "-ID")   { cfg.initial_depth = atoi(args[++i].c_str()); }
		else if (args[i] == "-NR")   { cfg.n_refine      = atoi(args[++i].c_str()); }
		else if (args[i] == "-NU")   { cfg.n_unrefine    = atoi(args[++i].c_str()); }
	}
	return cfg;
}


/////////////////////////////////////////////////////////////////
/// Phase: drop mesh elements that no longer have active dof.
/////////////////////////////////////////////////////////////////
void remove_unsupported_elements(Mesh_t& mesh, Handler_t& handler) {
	GUTIL_TIMER("remove mesh elements with no dofs");
	std::cout << mesh << "\n";

	mesh.request_unrefine([&handler](MeshElem_t el){
		return handler.get_active_dofs_quasi_hierarchical(el).empty();
	});

	mesh.process_unrefine();
	
	for (DOF_t dof : handler.active_dofs) {
		if (!mesh.is_conformal(DofVert_t{dof.key})) {
			GUTIL_ERROR(dof, " is not conformal");
		}
	}

	std::cout << mesh << "\n";
}

/////////////////////////////////////////////////////////////////
/// Phase: evaluate the field at every mesh vertex and save results to disk.
/////////////////////////////////////////////////////////////////
void evaluate_and_save(Mesh_t& mesh, Handler_t& handler, CoefHandler_t& coef_handler, const TestConfig& cfg) {
	mesh.collect_vertices();
	GUTIL_ASSERT(mesh.is_current());
	GUTIL_ASSERT(handler.is_current());

	mesh.set_encoded(false);
	mesh.sort_elements_by_color();

	std::vector<Scalar_t> scalar_vals(mesh.n_vertices());
	{
		GUTIL_LOG("Final mesh size: ", mesh.n_elements(), " elements and ", mesh.n_vertices(), " vertices");
		GUTIL_LOG("Final number of dofs: ", handler.n_dofs());
		GUTIL_TIMER("evaluating field at mesh vertices");

		scalar_vals = coef_handler.evaluate(0, mesh.vertex_begin(), mesh.vertex_end());
	}

	GUTIL_LOG("saving file");
	const std::string filename = cfg.test_name + "_dof.vtk";
	mesh.save_as_binary(filename);

	auto pt_field_lookup = GV::make_index_lookup<float>(
			[&](uint64_t idx){ return (float) scalar_vals[idx]; }, "scalar_field");

	auto pt_coef_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) {
				auto d_vtx = handler.get_dof_vertex(vtx);
				return d_vtx.exists() && handler.is_active_stable(DOF_t{d_vtx}) ? (int32_t)coef_handler.coefs[0][handler.global_number(DOF_t{d_vtx})] : -1;
			}, "scalar_coef");

	auto pt_dof_active_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) {
				auto d_vtx = handler.get_dof_vertex(vtx);
				return d_vtx.exists() && handler.is_active_stable(DOF_t{d_vtx}) ? (int32_t)handler.global_number(DOF_t{d_vtx}) : -1;
			}, "active_dof_index");

	auto pt_dof_refinable_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) {
				auto d_vtx = handler.get_dof_vertex(vtx);
				int val = -1;
				if (d_vtx.exists() && handler.is_active_stable(DOF_t{d_vtx}) ) {
					val = 10*(int)handler.is_refinable(DOF_t{d_vtx}) + (int)handler.is_unrefinable(DOF_t{d_vtx});
				}
				return val;
			}, "dof_ref_unref");

	auto pt_dof_key_lookup = GV::make_feature_lookup<MeshVert_t>(
			[&](MeshVert_t vtx) {
				auto d_vtx = handler.get_dof_vertex(vtx);
				return d_vtx.exists() && handler.is_active_stable(DOF_t{d_vtx}) ?
							std::array<int32_t,4>{(int32_t)d_vtx.depth(), (int32_t)d_vtx.i(), (int32_t)d_vtx.j(), (int32_t)d_vtx.k()} :
							std::array<int32_t,4>{-1,-1,-1,-1};
			}, "dof_key");

	auto pt_vtx_key_lookup = GV::make_feature_lookup<MeshVert_t>(
		[&](MeshVert_t vtx) {
			return std::array<int32_t,4>{(int32_t)vtx.depth(), (int32_t)vtx.i(), (int32_t)vtx.j(), (int32_t)vtx.k()};
		}, "viewed_vtx_key");

	auto el_dof_count_lookup = GV::make_feature_lookup<MeshElem_t>(
			[&](MeshElem_t el) {
				std::vector<DOF_t> dofs;
				std::vector<size_t> idxs;
				handler.get_active_dofs_full_hierarchical(el, dofs, idxs);
				return dofs.size();
			}, "n_active_dofs");

	auto el_dijkm_lookup = GV::make_feature_lookup<MeshElem_t>(
			[&](MeshElem_t el) {
				return std::array<int32_t,5>{(int32_t)el.depth(), (int32_t)el.i(), (int32_t)el.j(), (int32_t)el.k(), (int32_t)el.depth_linear_index()};

			}, "d_ijk_morton");

	auto el_color_lookup = GV::make_feature_lookup<MeshElem_t>(
			[&](MeshElem_t el) {
				return (uint16_t)el.color();
			}, "color54");

	auto el_idx_lookup = GV::make_index_lookup<uint64_t>(
			[&](uint64_t idx) {
				return idx;
			}, "element_number");

	mesh.append_point_data_field_binary(filename, "point",
			pt_dof_active_lookup, pt_dof_key_lookup, pt_field_lookup, pt_vtx_key_lookup);
	mesh.append_cell_data_field_binary(filename, "element",
			el_dof_count_lookup, el_idx_lookup, el_color_lookup, el_dijkm_lookup);
}

/////////////////////////////////////////////////////////////////
/// Top-level test driver
/////////////////////////////////////////////////////////////////
void test_dof_handler(const TestConfig& cfg) {
	Mesh_t mesh(domain, cfg.initial_depth + cfg.n_refine);
	mesh.set_depth(cfg.initial_depth);

	Handler_t handler(mesh);
	handler.init_dofs();

	CoefHandler_t coef_handler(handler);

	coef_handler.init_coefs(0,[](DOF_t dof){return 1.0;});

	//refine
	for (uint8_t i=0; i<cfg.n_refine; ++i) {
		GUTIL_TIMER("Refine ", i+1, "/", cfg.n_refine);
		handler.refine_quasi_hierarchical(mesh.element_begin(), mesh.element_end());
		mesh.process_refine();
		std::cout << handler << "\n";
		std::cout << mesh << "\n";
	}

	//unrefine
	for (uint8_t i=0; i<cfg.n_unrefine; ++i) {
		GUTIL_TIMER("Unrefine ", i+1, "/", cfg.n_refine);
		handler.unrefine_quasi_hierarchical(mesh.element_begin(), mesh.element_end());
		remove_unsupported_elements(mesh, handler);
		std::cout << handler << "\n";
		std::cout << mesh << "\n";
	}

	coef_handler.update_coefs();
	evaluate_and_save(mesh, handler, coef_handler, cfg);
}

int main(int argc, char* argv[]) {
	TestConfig cfg = parse_args(argc, argv);
	test_dof_handler(cfg);
	return 0;
}