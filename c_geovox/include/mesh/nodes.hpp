#ifndef NODES_H
#define NODES_H

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

#include <vector>
// #include <algorithm>
// #include <map>
// #include <set>
#include <iostream>
// #include <fstream>
// #include <string>
// #include <sstream>

// #include <stdexcept>

using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;



namespace GeoVox::mesh{
	class MeshNodeNode; //node of octree
	class MeshNode; //root node of octree is called MeshNode because it stores the FEM node locations
	using OctreeNode = GeoVox::util::OctreeNode<MeshNode, MeshNodeNode>;


	class MeshNodeNode : public OctreeNode{
		public:
			MeshNodeNode() : OctreeNode(), _npoints(0) {}

			MeshNodeNode(const Box& box, const long unsigned int ID, unsigned int (&ijk)[3], unsigned int depth, MeshNode* root) : 
				OctreeNode(box, ID, ijk, depth, root), _npoints(0) {}
			
			void make_children();
			long unsigned int npoints() const {return _npoints;} //number of points in children nodes //change to a stored quantity to make operator[] faster.

			void insert_point(const Point3 newpoint);
			void insert_point(std::vector<Point3> newpoints);

			Point3 operator[](const long unsigned int idx) const;

			long unsigned int find(const Point3& point) const;

			long unsigned int _npoints;
		protected:
			std::vector<Point3> _points;
	};


	class MeshNode : public MeshNodeNode {
	public:
		MeshNode() : MeshNodeNode(), _nleaves(1), max_points(16) {
			_root = this;
		}

		MeshNode(const std::vector<Point3>& list) : MeshNodeNode(), _nleaves(1), max_points(16) {
			_root = this;

			//get bounding box
			Point3 low = list[0];
			Point3 high = list[0];

			for (long unsigned int i=1; i<list.size(); i++){
				low = GeoVox::util::el_min(low, list[i]);
				high = GeoVox::util::el_max(high, list[i]);
			}

			box = Box(low, high);

			insert_point(list);
		}

		long unsigned int _nleaves;
		const long unsigned int max_points; //maximum number of points per leaf
	};

}
#endif