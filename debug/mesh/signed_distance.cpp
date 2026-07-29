#include "gutil.hpp"

// #include "mesh/voxel_mesh_structured.hpp"
#include "mesh/voxel_mesh_unstructured(NEW).hpp"
#include "diffuse_domain/signed_distance.hpp"

#include <string>
#include <vector>

using Point_t  = gutil::Point<3,double>;
using Box_t    = gutil::Box<3,double>;
using Sphere_t = gutil::Sphere<3,double>;
using Mesh_t   = GV::UnstructuredVoxelMesh<10>;
using Vert_t   = typename Mesh_t::VoxelVertex;
using Elem_t   = typename Mesh_t::VoxelElement;

int main(int argc, char* argv[]) {
	const int N = (argc>1) ? atoi(argv[1]) : 4;
	std::string filename = (argc > 2) ? argv[2] : "./testdata/sphere.txt";

	//read data and put into a signed distance octree
	std::vector<Sphere_t> list = gutil::read_spheres_from_file<3,double>(filename);

	Box_t box({0,0,0}, {2,2,2});
	GV::SignedDistanceSpheres<double,7> assembly(box);
	assembly.push_back_range(std::move(list));

	
	//create a mesh
	gutil::Logger::log("make mesh");
	Mesh_t mesh(box);
	
	for (int i=0; i<N; ++i) {
		gutil::LogTime t{"==== depth ", i+1, "/", N, "===="};
		{
			gutil::LogTime timer{"update_unstructured"};
			mesh.update_unstructured();
		}
	
		{
			gutil::LogTime timer{"refine mesh (setup)"};
			mesh.refine( [&](Elem_t el) { 
				Point_t pt = mesh.geo_coord(el.vertex(0));
				Point_t diag = mesh.geo_coord(el.vertex(7)) - pt;
				pt += 0.5*diag;

				double dist = assembly.signed_distance(pt);
				return dist*dist < 0.5*gutil::squared_norm(diag);
			});
		}
		
		{
			gutil::LogTime timer{"refine mesh (process)"};
			mesh.process_refine([&](Elem_t el) {
				return assembly.collides(Box_t{mesh.geo_coord(el.vertex(0)), mesh.geo_coord(el.vertex(7))});});
			// mesh.process_refine();
		}
	}




	gutil::Logger::log("make unstructured mesh");
	mesh.update_unstructured();

	{
		gutil::LogTime timer{"initialize colors"};
		mesh.init_color();
	}
	// {
	// 	gutil::LogTime timer{"synchronize depth field"};
	// 	mesh.synchronize_depth_field();
	// }
	

	//write to file and sample the signed distance
	gutil::Logger::log("make unstructured mesh vertices");
	mesh.collect_vertices();

	gutil::Logger::log("write topology to file");
	mesh.save_as_binary("signed_distance.vtk");

	auto sd_lookup = GV::make_feature_lookup<Vert_t>(
			[&assembly, &mesh](Vert_t vtx) {return assembly.signed_distance(mesh.geo_coord(vtx));},
			"signed_distance");

	auto heaviside_lookup = GV::make_feature_lookup<Vert_t>(
			[&assembly, &mesh](Vert_t vtx) {return assembly.heaviside(mesh.geo_coord(vtx), 0.1);},
			"heaviside");

	auto dirac_lookup = GV::make_feature_lookup<Vert_t>(
			[&assembly, &mesh](Vert_t vtx) {return assembly.dirac(mesh.geo_coord(vtx), 0.1);},
			"dirac");

	auto depth_lookup = GV::make_feature_lookup<Elem_t>(
			[](Elem_t el) {return el.depth();},
			"depth"
		);

	auto ijk_lookup = GV::make_feature_lookup<Elem_t>(
			[](Elem_t el) { return std::array<uint64_t,3>{el.i(), el.j(), el.k()}; },
			"ijk"
		);

	auto nbr_lookup = GV::make_feature_lookup<Elem_t>(
			[&mesh](Elem_t el) {
				int lo=99, hi=0;
				for (Elem_t nbr : mesh.neighbors(el)) {
					const int dd = static_cast<int>(nbr.depth()); 
					lo = std::min(lo, dd);
					hi = std::max(hi, dd);
				}
				return std::array<int,2>{lo,hi};
			},
			"neighbor_depth"
		);

	auto color_lookup = GV::make_feature_lookup<Elem_t>(
			[](Elem_t el) {
				return el.color();
			},
			"color"
		);

	auto depth_field_lookup = GV::make_feature_lookup<Elem_t>(
			[&](Elem_t el) {
				return mesh.read_depth(el);
			},
			"depth_field"
		);

	gutil::Logger::log("write details to file");
	mesh.append_point_data_field_binary("signed_distance.vtk", "sdf", sd_lookup, heaviside_lookup, dirac_lookup);
	mesh.append_cell_data_field_binary("signed_distance.vtk", "sdf", depth_lookup, ijk_lookup, nbr_lookup, color_lookup, depth_field_lookup);

	

	//get a layer and save its mesh and depth field
	// for (uint64_t dd=0; dd<=N; ++dd) {
	// 	gutil::LogTime timer{"saving layer " + std::to_string(dd)};

	// 	typename Mesh_t::S_Layer_t layer = mesh.get_layer(dd);
	// 	layer.set_all_active(true);
	// 	layer.update_unstructured_omp();
	// 	layer.collect_vertices_omp();

	// 	const std::string layer_filename = "mesh_layer_" + std::to_string(layer.depth) + ".vtk";
	// 	layer.save_as_binary(layer_filename);

	// 	auto layer_depth_field_lookup = GV::make_feature_lookup<Elem_t>(
	// 			[&](Elem_t el) {
	// 				return layer.read_depth(el);
	// 			},
	// 			"depth_field"
	// 		);

	// 	auto layer_active_lookup = GV::make_feature_lookup<Elem_t>(
	// 			[&](Elem_t el) {
	// 				return mesh.is_active(el);
	// 			},
	// 			"is_active"
	// 		);

	// 	layer.append_cell_data_field_binary(layer_filename, "debug_details", layer_depth_field_lookup, layer_active_lookup);
	// }


	gutil::Logger::log("done");
	return 0;
}






