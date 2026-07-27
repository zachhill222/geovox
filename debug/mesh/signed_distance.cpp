#include "gutil.hpp"

// #include "mesh/voxel_mesh_structured.hpp"
#include "mesh/voxel_mesh_unstructured(NEW).hpp"
#include "diffuse_domain/signed_distance.hpp"

#include <string>
#include <vector>

using Point_t  = gutil::Point<3,double>;
using Box_t    = gutil::Box<3,double>;
using Sphere_t = gutil::Sphere<3,double>;
using Mesh_t   = GV::UnstructuredVoxelMesh<8>;
using Vert_t   = typename Mesh_t::VoxelVertex;
using Elem_t   = typename Mesh_t::VoxelElement;

int main(int argc, char* argv[]) {
	std::string filename = (argc > 1) ? argv[1] : "./testdata/sphere.txt";

	//read data and put into a signed distance octree
	std::vector<Sphere_t> list = gutil::read_spheres_from_file<3,double>(filename);

	Box_t box({0,0,0}, {2,2,2});
	GV::SignedDistanceSpheres<double,7> assembly(box);
	assembly.push_back_range(std::move(list));

	
	//create a mesh
	gutil::Logger::log("make mesh");
	Mesh_t mesh(assembly.bbox(), 3);


	for (int i=0; i<4; ++i) {
		mesh.update_unstructured();
	
		gutil::Logger::log("refine mesh (setup)");
		mesh.refine( [&](Elem_t el) { 
			Point_t pt = mesh.geo_coord(el.vertex(0));
			Point_t diag = mesh.geo_coord(el.vertex(7)) - pt;
			pt += 0.5*diag;

			double dist = assembly.signed_distance(pt);
			return dist*dist < 0.5*gutil::squared_norm(diag);
			// return el.i() % 10 < 5;
		});

		gutil::Logger::log("refine mesh (process)");
		mesh.process_refine();
	}

	gutil::Logger::log("make unstructured mesh");
	mesh.update_unstructured();

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

	gutil::Logger::log("write details to file");
	mesh.append_point_data_field_binary("signed_distance.vtk", "sdf", sd_lookup, heaviside_lookup, dirac_lookup);

	gutil::Logger::log("done");
	return 0;
}






