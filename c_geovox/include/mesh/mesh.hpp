#ifndef MESH_H
#define MESH_H

#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;

namespace GeoVox::mesh{
	class Mesh{
	public:
		Mesh(long unsigned int nNodes, long unsigned int nElem){
			_node.reserve(nNodes);
			_elem2node.reserve(nElem);
			_vtkID.reserve(nElem);
		}

		Mesh(const std::vector<Point3> &node, const std::vector<std::vector<long unsigned int>> &elem2node, const std::vector<unsigned int> &vtkID): _node(node), _elem2node(elem2node), _vtkID(vtkID) {}

		Mesh(const std::vector<Point3> &node,
				const std::vector<std::vector<long unsigned int>> &elem2node,
				const std::vector<unsigned int> &vtkID,
				const std::vector<int> &nodeMarkers,
				const std::vector<int> &elemMarkers): _node(node), _elem2node(elem2node), _vtkID(vtkID), _nodeMarkers(nodeMarkers), _elemMarkers(elemMarkers) {}

		//mesh data
		inline long unsigned int nNodes() const {return _node.size();} //number of nodes
		inline long unsigned int nElems() const {return _elem2node.size();} //number of elements
		long unsigned int nTrackedNodes() const; //total nodes tracked in _elem2node
		Box box() const; //get bounding box for mesh


		//file io
		void saveas(const std::string filename) const; //save mesh to .vtk ASCII file

		//subdomains
		Mesh subdomain(const std::set<int>& eMarker) const;



		//essential data
		std::vector<Point3> _node; //point locations for each node
		std::vector<std::vector<long unsigned int>> _elem2node; //could be slow, but will allow different types of elements in the same mesh
		std::vector<unsigned int> _vtkID; //element type

		//optional data
		std::vector<int> _nodeMarkers; //track types of nodes (e.g., interior boundary, exterior boundary, ...)
		std::vector<int> _elemMarkers; //track types of elements (e.g., material type, subdomain numbers)
	};




}
#endif