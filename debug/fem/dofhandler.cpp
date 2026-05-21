#include "mesh/voxel_mesh_unstructured.hpp"
#include "fem/handlers/dofhandler_charms.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"

#include "mesh/vtk_file_io.hpp"

#include <cmath>
#include <fstream>
#include <cstdint>

using Mesh_t   = GV::UnstructuredVoxelMesh<10>;
using Elem_t   = Mesh_t::VoxelElement;
using Vert_t   = Mesh_t::VoxelVertex;
using DofKey_t = typename Vert_t::PeriodicVariant<0>;
using DOF_t    = GV::VoxelQ1<DofKey_t>;
using Basis_t  = GV::DofHandlerCharms<Mesh_t,DOF_t>;

int main(int argc, char* argv[])
{
	Mesh_t mesh({0,0,0}, {1,1,1});
	Basis_t basis(mesh);

	//activate mesh and basis to depth 4
	mesh.set_depth(3);
	basis.set_depth(3);

	//initialize a test scalar field
	std::vector<double> coefs(basis.n_dofs(), 0.0);
	basis.init_coefs_by_dof(coefs, [&mesh](const DOF_t dof) {
		const auto pt = mesh.geo_coord(static_cast<Vert_t>(dof.key));
		return std::sqrt(pt[0]*pt[0] + pt[1]*pt[1] + pt[2]*pt[2]);
	});

	// refine the mesh to depth 6 in the radial band (0.4, 0.6)
	for (uint64_t d=3; d<4; ++d) {
		std::vector<double> old_coefs = coefs;

		basis.refine(basis.curr_compressed_dofs());
		mesh.process_requests();
		
		basis.compress_dof_numbers();
		std::vector<double> new_coefs(basis.n_dofs(), 0.0);
		basis.update_coefs(new_coefs, old_coefs);
		coefs = std::move(new_coefs);
	}
	std::cout << "Done refining" << std::endl;

	//write the mesh structure to a file
	std::cout << "Writing to file" << std::endl;
	mesh.collect_vertices();
	mesh.save_as_ascii("dof_test.vtk");

	auto vert_vals = basis.interpolate_to_vertices(coefs, mesh.get_vertices());
	auto v_lookup = GV::make_index_lookup<double>([&](size_t idx){return vert_vals[idx];}, "function_value");
	mesh.append_point_data_field_ascii("dof_test.vtk", "point_data", v_lookup);



	return 0;
}