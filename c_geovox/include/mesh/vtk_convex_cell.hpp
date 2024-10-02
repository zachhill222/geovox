#ifndef VTK_CONVEX_CELL_H
#define VTK_CONVEX_CELL_H

#include "mesh/vtk_linear_elements.hpp"
#include "util/box.hpp"

#include <vector>

namespace GeoVox::mesh{

	//this is a specific use case of VTK_CONVEX_POINT_SET
	//nodes are stored in VTK_TRI_QUADRATIC_HEXAHEDRON order (sans the center node with index 26)
	//only the nodes relevant to the refined_faces are used.
	//pass global node numbers to the constructor in increasing local node number order
	//use the bool local_node_numbers to record which local nodes are used
	class OctCell : public GeoVox::mesh::Element<26> {
	public:
		OctCell(const std::vector<long unsigned int>& list, const bool (&local_nodes_numbers)[26]) : Element() {
			vertices.resize(26);
			long unsigned int n_idx = 0;
			for (int i=0; i<26; i++){
				used_nodes[i] = local_nodes_numbers[i];
				if (used_nodes[i]){
					vertices[i] = list[n_idx];
					n_idx += 1;
				}
			}
		}

		static const int vtkID = 41;
		bool used_nodes[26];
		int total_nodes() const;
		std::vector<long unsigned int> getface(const int face) const;

		inline GeoVox::mesh::Voxel tovoxel() const{
			return GeoVox::mesh::Voxel{vertices[0], vertices[1], vertices[3], vertices[2], vertices[4], vertices[5], vertices[7], vertices[6]};
		}

		inline GeoVox::mesh::Hexahedron tohex() const{
			return GeoVox::mesh::Hexahedron{vertices[0], vertices[1], vertices[2], vertices[3], vertices[4], vertices[5], vertices[6], vertices[7]};
		}
	};

}


#endif