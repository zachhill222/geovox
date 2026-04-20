#include "gutil.hpp"
#include "mesh/voxel_mesh.hpp"
#include "fem/dofs/dofhandler.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"

#include <cmath>
#include <fstream>
#include <cstdint>

using Mesh_t   = GV::VoxelMesh<10>;
using Elem_t   = Mesh_t::VoxelElement;
using Vert_t   = Mesh_t::VoxelVertex;
using DofKey_t = GV::VoxelVertexKey<11,1,0>;
using DOF_t    = GV::VoxelQ1<DofKey_t>;
using Basis_t  = GV::DofHandler<Mesh_t,DOF_t>;

int main(int argc, char* argv[])
{
	Mesh_t mesh({0,0,0}, {1,1,1});
	Basis_t basis(mesh);

	//activate mesh and basis to depth 4
	mesh.set_depth(3);
	basis.set_depth(3);

	//collect dof numbers for the active basis
	basis.compress_dof_numbers();

	//initialize a test scalar field
	std::vector<double> coefs(basis.n_dofs(), 0.0);
	basis.init_coefs_by_dof(coefs, [&mesh](const DOF_t dof) {
		const auto pt = mesh.ref2geo(static_cast<Vert_t>(dof.key));
		return std::sqrt(pt[0]*pt[0] + pt[1]*pt[1] + pt[2]*pt[2]);
	});

	// refine the mesh to depth 6 in the radial band (0.4, 0.6)
	for (uint64_t d=3; d<4; ++d) {
		basis.save_dof_list();
		std::vector<double> old_coefs = coefs;

		basis.refine_depth<true>(d, [&mesh](Vert_t vtx) {
			const auto pt = mesh.ref2geo(vtx);
			return pt[0] < 0.15;
		});
		mesh.process_request_active();
		mesh.process_request_deactive();

		basis.compress_dof_numbers();
		std::vector<double> new_coefs(basis.n_dofs(), 0.0);
		basis.update_coefs(old_coefs, new_coefs);
		coefs = std::move(new_coefs);
	}
	std::cout << "Done refining" << std::endl;

	//write the mesh structure to a file
	std::cout << "Writing to file" << std::endl;
	std::ofstream file("charmsQ1.vtk");
	const uint64_t n_verts = mesh.write_unstructured_vtk(file);

	file << "POINT_DATA " << n_verts << "\n";
	auto vert_vals = basis.interpolate_to_vertices(coefs, n_verts);
	mesh.append_unstructured_point_data_vtk(file,
					"SCALARS val float 1\nLOOKUP_TABLE default", 
					n_verts, 
					[&vert_vals](Vert_t vtx){return vert_vals[vtx.linear_index()];});



	return 0;
}