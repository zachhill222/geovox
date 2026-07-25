#include "gutil.hpp"

// #include "mesh/voxel_mesh_structured.hpp"
#include "mesh/voxel_mesh_unstructured(NEW).hpp"
#include "diffuse_domain/signed_distance.hpp"

#include <string>
#include <vector>

using Point_t = gutil::Point<3,double>;
using Box_t = gutil::Box<3,double>;
using Sphere_t = gutil::Sphere<3,double>;

int main(int argc, char* argv[]) {
	std::string filename = (argc > 1) ? argv[1] : "./testdata/sphere.txt";

	//read data and put into a signed distance octree
	std::vector<Sphere_t> list = gutil::read_spheres_from_file<3,double>(filename);

	Box_t box({0,0,0}, {2,2,2});
	GV::SignedDistanceSpheres<double,7> assembly(box);
	assembly.push_back_range(std::move(list));

	
	//create a mesh
	gutil::Logger::log("make mesh");
	// GV::StructuredVoxelMesh<8> mesh(assembly.bbox(),7);
	GV::UnstructuredVoxelMesh<8> mesh(assembly.bbox(), 7);
	// mesh.set_mask(true);
	gutil::Logger::log("make unstructured mesh");
	mesh.update_unstructured();
	auto& u_mesh = mesh;
	using Vert_t = typename decltype(mesh)::VoxelVertex;

	//write to file and sample the signed distance
	gutil::Logger::log("make unstructured mesh vertices");
	u_mesh.collect_vertices();

	gutil::Logger::log("write topology to file");
	u_mesh.save_as_binary("signed_distance.vtk");

	auto sd_lookup = GV::make_feature_lookup<Vert_t>(
			[&assembly, &u_mesh](Vert_t vtx) {return assembly.signed_distance(u_mesh.geo_coord(vtx));},
			"signed_distance");

	auto heaviside_lookup = GV::make_feature_lookup<Vert_t>(
			[&assembly, &u_mesh](Vert_t vtx) {return assembly.heaviside(u_mesh.geo_coord(vtx), 0.1);},
			"heaviside");

	auto dirac_lookup = GV::make_feature_lookup<Vert_t>(
			[&assembly, &u_mesh](Vert_t vtx) {return assembly.dirac(u_mesh.geo_coord(vtx), 0.1);},
			"dirac");

	gutil::Logger::log("write details to file");
	u_mesh.append_point_data_field_binary("signed_distance.vtk", "sdf", sd_lookup, heaviside_lookup, dirac_lookup);

	gutil::Logger::log("done");
	return 0;
}






