#ifndef OCTREE_VISUALIZE_H
#define OCTREE_VISUALIZE_H

#include "mesh/mesh.hpp"
#include "mesh/nodes.hpp"
#include "util/octree.hpp"

#include <vector>

using MeshNode = GeoVox::mesh::MeshNode;

template<typename root_t, typename node_t, typename data_t>
using OctreeNode = GeoVox::util::OctreeNode<root_t, node_t, data_t>;

namespace GeoVox::util{
	template<typename root_t, typename node_t, typename data_t>
	GeoVox::mesh::Mesh visualize_octree_structure(OctreeNode<root_t, node_t, data_t> const* octree){
		GeoVox::mesh::Mesh mesh;

		//get leaf box nodes
		mesh._node.box = octree->box;
		get_verticies(mesh._node, octree);

		//get leafs as elements
		mesh._node.set_ndata();
		get_elements(mesh._elem2node, mesh._elemMarkers, mesh._node, octree);
		std::vector<unsigned int> vtkElemID(mesh.nElems(), 11);
		mesh._vtkElemID = vtkElemID;

		return mesh;
	}


	template<typename root_t, typename node_t, typename data_t>
	void get_verticies(MeshNode &nodes, OctreeNode<root_t, node_t, data_t> const* octree){
		// std::cout << "node(" << octree->ID << "): size=" << octree->size() << std::endl;

		if (octree->isdivided()){
			for (int c_idx=0; c_idx<8; c_idx++){
				get_verticies(nodes, octree->child_const(c_idx));
			}
			return;
		}

		for (int v_idx=0; v_idx<8; v_idx++){
			// std::cout << "\tinsert vertex " << octree->box[v_idx].tostring() << std::endl;;
			nodes.insert_data(octree->box[v_idx]);
		}
	}


	template<typename root_t, typename node_t, typename data_t>
	void get_elements(std::vector<std::vector<long unsigned int>> &elem2node, std::vector<int> &elemMarkers, const MeshNode &nodes, OctreeNode<root_t, node_t, data_t> const* octree){
		if (octree->isdivided()){
			for (int c_idx=0; c_idx<8; c_idx++){
				get_elements(elem2node, elemMarkers, nodes, octree->child_const(c_idx));
			}
			return;
		}

		std::vector<long unsigned int> this_elem;
		this_elem.reserve(8);
		for (int v_idx=0; v_idx<8; v_idx++){
			this_elem.push_back(nodes.find(octree->box[v_idx]));
		}

		elem2node.push_back(this_elem);
		elemMarkers.push_back(octree->size());
	}
}


#endif