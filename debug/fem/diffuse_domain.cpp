#include "gutil.hpp"
#include "geovox.hpp"

#include <iostream>
#include <string>

#ifndef GV_TEST_DOF_PERIOD
	#define GV_TEST_DOF_PERIOD 7
#endif


/////////////////////////////////////////////////////////////////
/// Aliases
/////////////////////////////////////////////////////////////////
using Scalar_t      = double;
using Point_t       = gutil::Point<3,Scalar_t>;
using Box_t         = gutil::Box<3,Scalar_t>;
using Sphere_t      = gutil::Sphere<3,Scalar_t>;
using Assembly_t    = GV::SignedDistanceSpheres<Scalar_t,GV_TEST_DOF_PERIOD>;
using MeshHandler_t = GV::DiffuseDomainMeshHandler<Assembly_t>;

using Mesh_t 		= typename MeshHandler_t::Mesh_t;
using Elem_t 		= typename MeshHandler_t::Mesh_t::Elem_t;
using Vert_t		= typename MeshHandler_t::Mesh_t::Vert_t;

using DOF_t         = GV::Keys::DOFS::VoxelQ1<GV_TEST_DOF_PERIOD>;
using DofHandler_t  = GV::DofHandler<Mesh_t,DOF_t>;
using DofVert_t     = typename DofHandler_t::DofVert_t;
using DofElem_t     = typename DofHandler_t::DofElem_t;

using CoefHandler_t = GV::CoefHandler<DofHandler_t,Scalar_t,1>;

using InteriorWeight_t	= GV::AssemblyPhaseFieldWeight<true,Assembly_t>;
using ExteriorWeight_t	= GV::AssemblyPhaseFieldWeight<false,Assembly_t>;
using BKernel_t      	= GV::L2BilinearKernel<true,true>;
using BilinearForm_t    = GV::BilinearForm<4, Scalar_t, DofHandler_t, DofHandler_t, BKernel_t, InteriorWeight_t>;
using LKernel_t      	= GV::L2LinearKernel<true>;
using LinearForm_t      = GV::LinearForm<4,Scalar_t,DofHandler_t,LKernel_t,ExteriorWeight_t>;

inline constexpr Box_t domain{ {-1.1,-1.1,-1.1},
							   { 1.1, 1.1, 1.1} };
inline constexpr Scalar_t interior_exact{4.18879020479};
inline constexpr Scalar_t exterior_exact{domain.sidelength().prod() - interior_exact};

/////////////////////////////////////////////////////////////////
/// Runtime test configuration (from argv)
/////////////////////////////////////////////////////////////////
struct TestConfig {
	std::string test_name     	= "diffuse_domain";
	std::string file_name		= "sphere.txt";
	size_t      initial_depth 	= 3;			// initial mesh depth
	uint8_t     n_refine      	= 3;     	 	// number of refinements
};

TestConfig parse_args(int argc, char* argv[]) {
	TestConfig cfg;
	std::vector<std::string> args(argv, argv+argc);
	for (size_t i=0; i<args.size(); ++i) {
		if 		(args[i] == "-name") { cfg.test_name     = args[++i]; 				}
		else if (args[i] == "-file") { cfg.file_name     = args[++i]; 				}
		else if (args[i] == "-ID")   { cfg.initial_depth = atoi(args[++i].c_str()); }
		else if (args[i] == "-NR")   { cfg.n_refine      = atoi(args[++i].c_str()); }
	}
	return cfg;
}




int main(int argc, char* argv[]) {
	//read user args
	auto cfg = parse_args(argc, argv);
	
	//build the assembly and initialize the mesh
	MeshHandler_t mesh_handler(domain, cfg.initial_depth + cfg.n_refine);
	mesh_handler.build_assembly(cfg.file_name);
	mesh_handler.mesh.set_depth(cfg.initial_depth);
	
	//initialize the dofs and coefficients
	DofHandler_t  d_handler(mesh_handler.mesh);
	d_handler.init_dofs();

	CoefHandler_t c_handler(d_handler);
	c_handler.init_coefs(0,[](auto dof){return Scalar_t{1};});

	//link the diffuse domain weights to the assembly
	InteriorWeight_t::SetAssembly(mesh_handler.assembly);


	//refine the mesh near the interface and measure the volume
	{
		for (uint8_t r=0; r<cfg.n_refine; ++r) {
			std::cout << "\n";
			GUTIL_TIMER("Refinement ", r+1, "/", cfg.n_refine);

			Scalar_t eps = 3.0 * mesh_handler.min_element_size();
			InteriorWeight_t::SetEps(eps);

			//note that refine requests can be made via const reference
			const Mesh_t& mesh = mesh_handler.mesh;

			// auto collides = [eps,&mesh_handler, &mesh](Elem_t el) {
			// 	Box_t cell{mesh.geo_coord(el.vertex(0)), mesh.geo_coord(el.vertex(7))};
			// 	return mesh_handler.assembly.collides(cell);
			// };

			auto near_surface = [eps,&mesh_handler, &mesh](Elem_t el) {
				auto diag = mesh.geo_coord(el.vertex(7))-mesh.geo_coord(el.vertex(0));
				auto sd = mesh_handler.assembly.signed_distance(mesh_handler.mesh.geo_center(el));
				return std::abs(sd) < 1.5*gutil::norm2(diag);
			};

			std::vector<Elem_t> ref_elems = mesh_handler.mesh.select_elements(near_surface);

			d_handler.refine_quasi_hierarchical(ref_elems);
			c_handler.prolong_coefs();

			//process the refinement request, you may pass a predicate to only activate
			//elements that satisfy it.
			mesh_handler.mesh.process_refine( [](Elem_t el) {
					return true;
				});

			GUTIL_LOG("Test volumes: eps= ", eps);

			//Test the exterior volume
			{
				LinearForm_t l_form(d_handler);
				Scalar_t approx = l_form.evaluate_form(c_handler.get_coefs(0));
				GUTIL_LOG("Exteror: exact=", exterior_exact, " approx=", approx, " err=", std::abs(approx-exterior_exact));
			}

			//Test the interior volume
			{
				BilinearForm_t b_form(d_handler);
				std::vector<Scalar_t> vec(d_handler.n_dofs(), 0);

				b_form.mat_vec_multiply_accumulate(GV::as_span(vec), c_handler.get_coefs(0));

				Scalar_t approx = gutil::dot_product_reduce<Scalar_t>(vec, c_handler.get_coefs(0));
				GUTIL_LOG("Interor: exact=", interior_exact, " approx=", approx, " err=", std::abs(approx-interior_exact));
			}
		}
	}

	//the vertices need to be explicitly stored to write as a vtk
	{
		GUTIL_TIMER("Collect mesh vertices");
		mesh_handler.mesh.collect_vertices();
	}


	//save the mesh
	{	
		const std::string filename = cfg.test_name + ".vtk";
		const Assembly_t& assembly = mesh_handler.assembly;
		const Mesh_t&	  mesh 	   = mesh_handler.mesh;

		GUTIL_TIMER("Save mesh as ", filename);
		std::cout << mesh_handler.mesh << std::endl;
		
		mesh_handler.mesh.save_as_binary(filename);

		Scalar_t eps = 3.0 * mesh_handler.min_element_size();
		
		auto sd_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {
				return assembly.signed_distance(mesh.geo_coord(vtx));},
			"signed_distance");

		auto heaviside_lookup = GV::make_feature_lookup<Vert_t>(
				[&](Vert_t vtx) {return assembly.heaviside(mesh.geo_coord(vtx), eps);},
				"heaviside");

		auto heaviside_lookup = GV::make_feature_lookup<Vert_t>(
				[&](Vert_t vtx) {return assembly.heaviside(mesh.geo_coord(vtx), eps);},
				"heaviside");

		auto heaviside_grad_lookup = GV::make_feature_lookup<Vert_t>(
				[&](Vert_t vtx) {
					return assembly.heaviside_grad(mesh.geo_coord(vtx), eps);},
				"heaviside_grad");

		auto depth_lookup = GV::make_feature_lookup<Elem_t>(
				[](Elem_t el) {return el.depth();},
				"depth"
			);

		auto ijk_lookup = GV::make_feature_lookup<Elem_t>(
				[](Elem_t el) { return std::array<uint64_t,3>{el.i(), el.j(), el.k()}; },
				"ijk"
			);

		auto el_dijkm_lookup = GV::make_feature_lookup<Elem_t>(
				[&](Elem_t el) {
					return std::array<int32_t,5>{(int32_t)el.depth(), (int32_t)el.i(), (int32_t)el.j(), (int32_t)el.k(), (int32_t)el.depth_linear_index()};

				}, "d_ijk_morton");

		auto el_color_lookup = GV::make_feature_lookup<Elem_t>(
				[&](Elem_t el) {
					return (uint16_t)el.color();
				}, "color54");

		auto el_idx_lookup = GV::make_index_lookup<uint64_t>(
				[&](uint64_t idx) {
					return idx;
				}, "element_number");

		mesh.append_point_data_field_binary(filename, "point_data", sd_lookup, heaviside_lookup, heaviside_lookup, heaviside_grad_lookup);
		mesh.append_cell_data_field_binary(filename, "cell_data", depth_lookup, el_dijkm_lookup, el_color_lookup, el_idx_lookup);
	}

	

	



}