#include "gutil.hpp"

#include "mesh/voxel_mesh_unstructured.hpp"
#include "geometry/signed_distance.hpp"

#include <string>

int main(int argc, char* argv[]) {
	std::string filename = (argc > 1) ? argv[1] : "./testdata/sphere.txt";

	//read data and put into a signed distance octree
	auto list = gutil::read_spheres_from_file<3,double>(filename);
	GV::SignedDistanceSpheres<double> assembly(std::move(list));
	
	//create a mesh
	GV::UnstructuredVoxelMesh<10> mesh(assembly.bbox().low, assembly.bbox().high);
	using Vert_t = typename decltype(mesh)::VoxelVertex;

	mesh.set_depth(7);

	//write to file and sample the signed distance
	mesh.collect_vertices();
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

	mesh.append_point_data_field_binary("signed_distance.vtk", "sdf", sd_lookup, heaviside_lookup, dirac_lookup);

	return 0;
}






