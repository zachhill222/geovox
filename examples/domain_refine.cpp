#include "gutil.hpp"
#include "geovox.hpp"

#include <iostream>
#include <string>

#ifndef GV_TEST_DOMAIN_PERIOD
	#define GV_TEST_DOMAIN_PERIOD 7
#endif


/////////////////////////////////////////////////////////////////
/// Aliases
/////////////////////////////////////////////////////////////////
using Scalar_t      = double;
using Point_t       = gutil::Point<3,Scalar_t>;
using Box_t         = gutil::Box<3,Scalar_t>;
using Sphere_t      = gutil::Sphere<3,Scalar_t>;
using Assembly_t    = GV::SignedDistanceSpheres<Scalar_t,GV_TEST_DOMAIN_PERIOD>;
using MeshHandler_t = GV::DiffuseDomainMeshHandler<Assembly_t>;

using Mesh_t 		= typename MeshHandler_t::Mesh_t;
using Elem_t 		= typename MeshHandler_t::Mesh_t::Elem_t;
using Vert_t		= typename MeshHandler_t::Mesh_t::Vert_t;

/////////////////////////////////////////////////////////////////
/// Runtime test configuration (from argv)
/////////////////////////////////////////////////////////////////
struct TestConfig {
	std::string test_name     	= "domain_refine";
	std::string file_name		= "spheres.txt";
	size_t      initial_depth 	= 3;			// initial mesh depth
	Scalar_t    tol 		   	= 0.25;  	 	// signed-distance tolerance for boundary refinement
	uint8_t     n_refine      	= 3;     	 	// number of refinements
};

TestConfig parse_args(int argc, char* argv[]) {
	TestConfig cfg;
	std::vector<std::string> args(argv, argv+argc);
	for (size_t i=0; i<args.size(); ++i) {
		if 		(args[i] == "-name") { cfg.test_name     = args[++i]; 				}
		else if (args[i] == "-file") { cfg.file_name     = args[++i]; 				}
		else if (args[i] == "-ID")   { cfg.initial_depth = atoi(args[++i].c_str()); }
		else if (args[i] == "-TOL")  { cfg.tol 			 = atof(args[++i].c_str()); }
		else if (args[i] == "-NR")   { cfg.n_refine      = atoi(args[++i].c_str()); }
	}
	return cfg;
}




int main(int argc, char* argv[]) {
	GUTIL_TIMER("Building mesh");
	
	//read user args
	auto cfg = parse_args(argc, argv);
	
	//build the assembly and initialize the mesh
	Box_t domain{{-1,-1,-1}, {1,1,1}};	
	MeshHandler_t mesh_handler(domain, cfg.initial_depth + cfg.n_refine);
	mesh_handler.build_assembly(cfg.file_name);
	mesh_handler.mesh.set_depth(cfg.initial_depth);
	


	//refine the mesh near the interface
	{
		GUTIL_TIMER("Refine mesh");
		for (uint8_t r=0; r<cfg.n_refine; ++r) {
			GUTIL_TIMER("   Refinement ", r+1, "/", cfg.n_refine);

			Scalar_t eps = 0.666666*mesh_handler.min_element_size();
			Scalar_t tol = mesh_handler.min_element_size() + cfg.tol;
			
			//note that refine requests can be made via const reference
			const Mesh_t& mesh = mesh_handler.mesh;

			mesh.request_refine([eps,tol,&mesh_handler, &mesh](Elem_t el) {
				Point_t el_center	= mesh.geo_center(el);
				Scalar_t sdf 		= mesh_handler.assembly.signed_distance(el_center);
				return std::abs(sdf) < tol;
			});

			//process the refinement request, you may pass a predicate to only activate
			//elements that satisfy it.
			mesh_handler.mesh.process_refine( [](Elem_t el) {
					return true;
				});
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

		Scalar_t eps = 0.666666*mesh_handler.min_element_size();
		
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
				[&](Vert_t vtx) {return assembly.heaviside_grad(mesh.geo_coord(vtx), eps);},
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