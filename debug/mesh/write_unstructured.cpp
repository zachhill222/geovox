#include "mesh/voxel_mesh.hpp"
#include "mesh/voxel_mesh_unstructured.hpp"
#include "mesh/vtk_file_io.hpp"

using Point_t = GV::Point<3,double>;
using Mesh_t  = GV::VoxelMesh<6>;
using Elem_t  = Mesh_t::VoxelElement;
using Vert_t  = Mesh_t::VoxelVertex;

int main(int argc, char* argv[])
{
	//get depth
	size_t d=3;
	if (argc>1) {
		const size_t dd = atoi(argv[1]);
		if (dd>=0 and dd<=Mesh_t::MAX_DEPTH) {
			d = dd;
		}
	}
	
	//build test mesh
	Point_t low{-1,-1,-1};
	Point_t high{1,1,1};
	Mesh_t  mesh(low, high);

	//print all elements at a depth
	auto pred = [&](Elem_t el) {
		auto v1 = el.vertex(0);
		auto v2 = el.vertex(7);
		auto center = 0.5*(mesh.ref2geo(v1)+mesh.ref2geo(v2));
		return GV::squaredNorm(center) < 0.25;
	};

	mesh.set_depth(1);
	mesh.refine_to_depth(Elem_t{1,0,0,0},d,pred);
	mesh.save_unstructured_mesh("structured.vtk");


	//convert to an unstructured mesh
	GV::UnstructuredVoxelMesh<Mesh_t::MAX_DEPTH> u_mesh(mesh);
	u_mesh.collect_vertices();
	u_mesh.color();
	u_mesh.save_as_ascii("unstructured.vtk");
	u_mesh.save_as_binary("unstructured_binary.vtk");

	auto clr_lookup = GV::make_lookup<Elem_t>([](Elem_t el){return el.color();}, "color");
	auto idx_lookup = GV::make_lookup<Elem_t>([](Elem_t el){return el.linear_index();}, "linear_index");
	auto ijk_lookup = GV::make_lookup<Elem_t>([](Elem_t el){return std::array<uint64_t,3>{el.i(), el.j(), el.k()};}, "ijk");
	auto d_lookup = GV::make_lookup<Elem_t>([](Elem_t el){return el.depth();}, "depth");

	GV::append_cell_data_field_vtk<decltype(u_mesh),true>("unstructured.vtk", u_mesh, "cell_test", clr_lookup, idx_lookup, ijk_lookup, d_lookup);
	GV::append_cell_data_field_vtk<decltype(u_mesh),false>("unstructured_binary.vtk", u_mesh, "cell_test", clr_lookup, idx_lookup, ijk_lookup, d_lookup);

	auto xyz_lookup = GV::make_lookup<Vert_t>([](Vert_t vtx){return std::array<double,3>{vtx.x(), vtx.y(), vtx.z()};}, "xyz");
	auto fun_val    = GV::make_lookup<Vert_t>([&u_mesh](Vert_t vtx){
		auto coord = u_mesh.geo_coord(vtx);
		return coord[0]*coord[1]*coord[2];}, "function");

	GV::append_point_data_field_vtk<decltype(u_mesh),true>("unstructured.vtk", u_mesh, "point_test", xyz_lookup, fun_val);
	GV::append_point_data_field_vtk<decltype(u_mesh),false>("unstructured_binary.vtk", u_mesh, "point_test", xyz_lookup, fun_val);

	return 0;
}