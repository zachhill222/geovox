#ifndef MESH_H
#define MESH_H

#include "util/point.hpp"
#include "util/box.hpp"

#include "mesh/nodes.hpp"
#include "mesh/vtk_linear_elements.hpp"

#include <vector>
#include <algorithm>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <stdexcept>

using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;
using MeshNode = GeoVox::mesh::MeshNode;

namespace GeoVox::mesh{
	class Mesh{
	public:
		Mesh() {}

		Mesh(long unsigned int nNodes, long unsigned int nElems){
			// _node.reserve(nNodes);
			_elem2node.reserve(nElems);
			_vtkElemID.reserve(nElems);
			_node2elem.reserve(nNodes);
		}

		Mesh(const MeshNode &node, const std::vector<std::vector<long unsigned int>> &elem2node, const std::vector<unsigned int> &vtkElemID): _node(node), _elem2node(elem2node), _vtkElemID(vtkElemID) {
			make_node2elem();
		}

		Mesh(const Mesh& other): _node(other._node), _elem2node(other._elem2node), _node2elem(other._node2elem), _vtkElemID(other._vtkElemID), 
									_face2node(other._face2node), _vtkFaceID(other._vtkFaceID),
										_nodeMarkers(other._nodeMarkers), _elemMarkers(other._elemMarkers) {}

		Mesh(const std::vector<Point3> &node,
				const std::vector<std::vector<long unsigned int>> &elem2node,
				const std::vector<unsigned int> &vtkElemID,
				const std::vector<int> &nodeMarkers,
				const std::vector<int> &elemMarkers): _node(node), _elem2node(elem2node), _vtkElemID(vtkElemID), _nodeMarkers(nodeMarkers), _elemMarkers(elemMarkers) {}

		//mesh data
		inline long unsigned int nNodes() const {return _node.size();} //number of nodes
		inline long unsigned int nElems() const {return _elem2node.size();} //number of elements
		inline long unsigned int nFaces() const {return _face2node.size();} //number of faces (boundary)
		long unsigned int nTrackedNodes() const; //total nodes tracked in _elem2node
		Box box() const; //get bounding box for mesh

		void make_node2elem();
		void make_boundary();

		//file io
		void saveas(const std::string filename) const; //save mesh to .vtk ASCII file

		//subdomains
		Mesh mesh_subdomain(const std::set<int>& eMarker) const;
		// Mesh boundary_mesh(const std::set<int>& eMarker) const;
		// Mesh extract_boundary_mesh();

		//essential data
		// std::vector<Point3> _node; //point locations for each node
		MeshNode _node;
		std::vector<std::vector<long unsigned int>> _elem2node; //could be slow, but will allow different types of elements in the same mesh
		std::vector<std::vector<long unsigned int>> _node2elem;
		std::vector<unsigned int> _vtkElemID; //element type

		//boundary data
		std::vector<std::vector<long unsigned int>> _face2node;
		std::vector<unsigned int> _vtkFaceID;

		//extra data
		std::vector<int> _nodeMarkers; //track types of nodes (e.g., interior boundary, exterior boundary, ...)
		std::vector<int> _elemMarkers; //track types of elements (e.g., material type, subdomain numbers)
	};
}
#endif