#include "gutil.hpp"
#include "geovox.hpp"

#ifndef GV_TEST_DOF_PERIOD
	#define GV_TEST_DOF_PERIOD 0
#endif


/////////////////////////////////////////////////////////////////
/// Aliases
/////////////////////////////////////////////////////////////////
using Scalar_t      = double;
using Point_t       = gutil::Point<3,Scalar_t>;
using Box_t         = gutil::Box<3,Scalar_t>;
using Sphere_t      = gutil::Sphere<3,Scalar_t>;

using Mesh_t 		= GV::UnstructuredVoxelMesh<Scalar_t>;
using Elem_t 		= typename Mesh_t::Elem_t;
using Vert_t		= typename Mesh_t::Vert_t;

using DOF_t         = GV::Keys::DOFS::VoxelQ1<GV_TEST_DOF_PERIOD>;
using DofHandler_t  = GV::CharmsHandlerTH<Mesh_t,DOF_t>;
using DofVert_t     = typename DofHandler_t::DofVert_t;
using DofElem_t     = typename DofHandler_t::DofElem_t;

using CoefHandler_t = GV::CoefHandler<DofHandler_t,Scalar_t,1>;

using Kernel_t      = GV::L2BilinearKernel<>;
// using Kernel_t      = GV::H1BilinearKernel<>;
using BilinearForm_t    = GV::BilinearForm<4, Scalar_t, DofHandler_t, DofHandler_t, Kernel_t>;

inline constexpr Box_t domain{ {-1,-1,-1},
							   { 1, 1, 1} };

inline constexpr Scalar_t exact{8};

/////////////////////////////////////////////////////////////////
/// Runtime test configuration (from argv)
/////////////////////////////////////////////////////////////////
struct TestConfig {
	std::string test_name     	= "bilinearform";
	size_t      initial_depth 	= 2;			// initial mesh depth
	uint8_t     n_refine      	= 1;     	 	// number of refinements
	uint8_t     n_unrefine      = 1;     	 	// number of unrefinements
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


void test_box_domain(Mesh_t& mesh, DofHandler_t& d_handler, CoefHandler_t& c_handler) {
	BilinearForm_t b_form(d_handler,d_handler);
	
	mesh.set_encoded(false);
	mesh.sort_elements_by_color();
	{
		std::vector<Scalar_t> y(d_handler.n_dofs(), 0);
		// std::vector<Scalar_t> ones(d_handler.n_dofs(), 1);
		std::span<const Scalar_t> coefs = c_handler.get_coefs(0);

		GUTIL_TIMER("Test bilinear form vector evaluation (colored)");
		// b_form.mat_vec_multiply_accumulate_colored(GV::as_span(y), coefs);
		// Scalar_t approx = gutil::dot_product_reduce<Scalar_t>(y,coefs);
		
		Scalar_t approx = b_form.evaluate(coefs,coefs);

		GUTIL_LOG("approx=", approx, " exact=", exact, " (error=", std::abs(exact-approx), ")");
	}
	mesh.sort_elements_by_depth();
	mesh.set_encoded(true);
}


void refine(TestConfig cfg, Mesh_t& mesh, DofHandler_t& d_handler, CoefHandler_t& c_handler) {
	for (uint8_t i=0; i<cfg.n_refine; ++i) {
		std::vector<Elem_t> elements;
		for (auto it = mesh.element_begin(); it!=mesh.element_end(); ++it) {
			if (mesh.geo_center(*it)[i%3] > Scalar_t{0}){
				elements.push_back(*it);
			}
		}

		d_handler.refine(elements);
		// d_handler.refine(mesh.element_begin(), mesh.element_end());
		mesh.process_refine();
		c_handler.prolong_coefs();
	}

}

void unrefine(TestConfig cfg, Mesh_t& mesh, DofHandler_t& d_handler, CoefHandler_t& c_handler) {
	for (uint8_t i=0; i<cfg.n_unrefine; ++i) {
		std::vector<Elem_t> elements;
		for (auto it = mesh.element_begin(); it!=mesh.element_end(); ++it) {
			if (mesh.geo_center(*it)[(i+1)%3] > Scalar_t{0}){
				elements.push_back(*it);
			}
		}
		
		d_handler.unrefine(elements);

		for (auto it = mesh.element_begin(); it!=mesh.element_end(); ++it) {
			if (d_handler.get_active_dofs_conformal(*it).empty()) {
				mesh.request_unrefine(*it);
			}
		}
		
		mesh.process_unrefine();
		c_handler.restrict_coefs();
	}
}

void save_mesh(const std::string& filename, Mesh_t& mesh, DofHandler_t& d_handler, CoefHandler_t& c_handler) {
	GUTIL_TIMER("saving mesh as ", filename);
	mesh.collect_vertices();
	GUTIL_ASSERT(mesh.is_current());
	GUTIL_ASSERT(d_handler.is_current());

	std::vector<Scalar_t> scalar_vals(mesh.n_vertices());
	scalar_vals = c_handler.evaluate(0, mesh.vertex_begin(), mesh.vertex_end());

	mesh.save_as_binary(filename);

	auto pt_field_lookup = GV::make_index_lookup<float>(
			[&](uint64_t idx){ return (float) scalar_vals[idx]; }, "scalar_field");
	auto pt_coef_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {
				auto d_vtx = d_handler.get_dof_vertex(vtx);
				return d_vtx.exists() && d_handler.is_active_stable(DOF_t{d_vtx}) ? 
							(float)c_handler.get_coefs(0)[d_handler.global_number(DOF_t{d_vtx})] : -1;
			}, "scalar_coef");
	auto pt_dof_key_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {
				auto d_vtx = d_handler.get_dof_vertex(vtx);
				return d_vtx.exists() && d_handler.is_active_stable(DOF_t{d_vtx}) ?
							std::array<int32_t,4>{(int32_t)d_vtx.depth(), (int32_t)d_vtx.i(), (int32_t)d_vtx.j(), (int32_t)d_vtx.k()} :
							std::array<int32_t,4>{-1,-1,-1,-1};
			}, "dof_key");
	auto el_dijkm_lookup = GV::make_feature_lookup<Elem_t>(
			[&](Elem_t el) {
				return std::array<int32_t,5>{(int32_t)el.depth(), (int32_t)el.i(), (int32_t)el.j(), (int32_t)el.k(), (int32_t)el.depth_linear_index()};

			}, "d_ijk_morton");

	mesh.append_point_data_field_binary(filename, "point", pt_field_lookup, pt_coef_lookup, pt_dof_key_lookup);
	mesh.append_cell_data_field_binary(filename, "cell", el_dijkm_lookup);
}



int main(int argc, char* argv[]) {
	TestConfig cfg = parse_args(argc, argv);

	Mesh_t mesh(domain, cfg.initial_depth+cfg.n_refine);
	mesh.set_depth(cfg.initial_depth);

	DofHandler_t d_handler(mesh);
	d_handler.init_dofs();

	CoefHandler_t c_handler(d_handler);
	c_handler.init_coefs();
	// c_handler.assign_coefs(0, [&](DOF_t dof) {return mesh.geo_coord(Vert_t{dof.key})[0];});
	c_handler.assign_coefs(0, [&](DOF_t dof) {return 1;});

	save_mesh(cfg.test_name + "_initial.vtk", mesh, d_handler, c_handler);
	std::cout << "\n\n";
	std::cout << mesh << d_handler;
	{
		GUTIL_TIMER("Uniform mesh");
		test_box_domain(mesh, d_handler, c_handler);
	}
	
	std::cout << "\n\n";
	refine(cfg, mesh, d_handler, c_handler);
	std::cout << mesh << d_handler;
	save_mesh(cfg.test_name + "_refined.vtk", mesh, d_handler, c_handler);
	{
		GUTIL_TIMER("Refined mesh");
		test_box_domain(mesh, d_handler, c_handler);
	}

	std::cout << "\n\n";
	unrefine(cfg, mesh, d_handler, c_handler);
	std::cout << mesh << d_handler;
	save_mesh(cfg.test_name + "_unrefined.vtk", mesh, d_handler, c_handler);
	{
		GUTIL_TIMER("Unrefined mesh");
		test_box_domain(mesh, d_handler, c_handler);
	}

	gutil::print_all_profiles();
	return 0;
}











