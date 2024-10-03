#ifndef NODES_H
#define NODES_H

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

#include <vector>
#include <iostream>

using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;



namespace GeoVox::mesh{
	class MeshNodeNode; //node of octree
	class MeshNode; //root node of octree is called MeshNode because it stores the FEM node locations
	using OctreeNode = GeoVox::util::OctreeNode<MeshNode, MeshNodeNode, Point3>;


	class MeshNodeNode : public OctreeNode{
		public:
			MeshNodeNode() : OctreeNode() {}

			MeshNodeNode(const Box& box, const long unsigned int ID, unsigned int (&ijk)[3], unsigned int depth, MeshNode* root) : 
				OctreeNode(box, ID, ijk, depth, root) {}
			
			bool data_valid(const Point3& point) const override;
	};


	class MeshNode : public MeshNodeNode {
	public:
		MeshNode() : MeshNodeNode(), _nleaves(1), _maxdepth(0), max_data_per_leaf(16) {
			_root = this;
		}

		MeshNode(const std::vector<Point3>& list) : MeshNodeNode(), _nleaves(1), _maxdepth(0), max_data_per_leaf(16) {
			_root = this;

			//get bounding box
			Point3 low = list[0];
			Point3 high = list[0];

			for (long unsigned int i=1; i<list.size(); i++){
				low = GeoVox::util::el_min(low, list[i]);
				high = GeoVox::util::el_max(high, list[i]);
			}

			box = Box(low, high);

			for (long unsigned int d_idx=0; d_idx<list.size(); d_idx++){
				insert_data(list[d_idx]);
			}
		}

		long unsigned int _nleaves;
		unsigned int _maxdepth;
		const long unsigned int max_data_per_leaf; //maximum number of points per leaf
	};

}
#endif