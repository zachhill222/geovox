#include "gutil.hpp"

#include "mesh/voxel_mesh_unstructured.hpp"
#include "diffuse_domain/signed_distance.hpp"

#include <string>
#include <vector>

using Point_t  = gutil::Point<3,double>;
using Box_t    = gutil::Box<3,double>;
using Sphere_t = gutil::Sphere<3,double>;
using Mesh_t   = GV::UnstructuredVoxelMesh<double>;
using Vert_t   = typename Mesh_t::Vert_t;
using Elem_t   = typename Mesh_t::Elem_t;
using Assem_t  = GV::SignedDistanceSpheres<double,1>;

int main(int argc, char* argv[]) {
	const int N = (argc>1) ? atoi(argv[1]) : 4;
	std::string filename = (argc > 2) ? argv[2] : "./testdata/periodic_spheres.txt";

	//read data and put into a signed distance octree
	std::vector<Sphere_t> list = gutil::read_spheres_from_file<3,double>(filename);

	Box_t box({-1,-1,-1}, {1,1,1});
	Assem_t assembly(box);
	assembly.push_back_range(std::move(list));
	if( assembly.size()==0 ) {GUTIL_ABORT("no spheres");}
	
	//create a mesh
	int id = 2; int max_d = id+N;
	gutil::Logger::log("make mesh with ", assembly.size(), " spheres with initial depth ", id, "/", max_d);
	Mesh_t mesh(box, max_d);
	mesh.set_depth(id);
	for (int i=0; i<N; ++i) {
		gutil::LogTime t{"==== refinement ", i+1, "/", N, "===="};
		{
			gutil::LogTime timer{"refine mesh (setup)"};
			mesh.request_refine( [&](Elem_t el) { 
				Point_t pt = mesh.geo_center(el);
				double dist = assembly.signed_distance(pt);
				return dist*dist < 0.5*gutil::squared_norm(mesh.el_size(el));
			});
		}
		
		{
			gutil::LogTime timer{"refine mesh (process)"};
			mesh.process_refine([&](Elem_t el) {
				return assembly.collides(Box_t{mesh.geo_coord(el.vertex(0)), mesh.geo_coord(el.vertex(7))});});
			// mesh.process_refine();
		}

		gutil::Logger::log("mesh has ", mesh.n_elements(), " elements");
	}

	gutil::Logger::log("make unstructured mesh");
	mesh.collect_elements();

	//write to file and sample the signed distance
	gutil::Logger::log("make unstructured mesh vertices");
	mesh.collect_vertices();

	gutil::Logger::log("write topology to file");
	mesh.save_as_binary("signed_distance.vtk");

	auto sd_lookup = GV::make_feature_lookup<Vert_t>(
			[&assembly, &mesh](Vert_t vtx) {
				return assembly.signed_distance(mesh.geo_coord(vtx));},
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

	auto depth_field_lookup = GV::make_feature_lookup<Elem_t>(
			[&](Elem_t el) {
				return mesh.read_depth_field(el);
			},
			"depth_field"
		);

	gutil::Logger::log("write details to file");
	mesh.append_point_data_field_binary("signed_distance.vtk", "sdf", sd_lookup, heaviside_lookup, dirac_lookup);
	mesh.append_cell_data_field_binary("signed_distance.vtk", "sdf", depth_lookup, ijk_lookup, nbr_lookup, depth_field_lookup);

	

	//get a slice and save its mesh and depth field
	gutil::LogTime timer{"saving slice"};
	uint8_t dd=4;

	Mesh_t slice = mesh.get_depth_slice_as_mesh(dd,dd);
	//mark all elements as active so we can view the depth field
	{
		auto lock = slice.begin_key_mask_unstable();

		uint64_t depth_start = Elem_t::elements_below_depth(dd);
		uint64_t depth_end = Elem_t::elements_below_depth(dd+1);
		std::span<uint8_t> mask = slice.get_mask_span_ref(depth_start,depth_end);

		GUTIL_SIMD()
		for (size_t i=0; i<mask.size(); ++i) {
			mask[i]|=Mesh_t::ACTIVE_BIT;
		}

		slice.end_key_mask_unstable();
	}


	slice.collect_elements();
	slice.collect_vertices();

	const std::string slice_filename = "mesh_slice.vtk";
	slice.save_as_binary(slice_filename);

	auto slice_depth_field_lookup = GV::make_feature_lookup<Elem_t>(
			[&](Elem_t el) {
				return slice.read_depth_field(el);
			},
			"depth_field"
		);

	auto slice_active_lookup = GV::make_feature_lookup<Elem_t>(
			[&](Elem_t el) {
				return mesh.is_active(el);
			},
			"is_active"
		);

	slice.append_cell_data_field_binary(slice_filename, "debug_details", slice_depth_field_lookup, slice_active_lookup);


	gutil::Logger::log("done");
	return 0;
}






