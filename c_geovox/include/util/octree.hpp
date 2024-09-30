#ifndef OCTREE_H
#define OCTREE_H

#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <map>

using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;

namespace GeoVox::util{

	template<typename root_t, typename node_t>
	class OctreeNode{
	public:
		OctreeNode() : box(Box()), ID(0), ijk{0,0,0}, _isdivided(false), _depth(0), _root(NULL) {}

		OctreeNode(const Box& box) : box(box), ID(0), ijk{0,0,0}, _isdivided(false), _depth(0), _root(NULL) {}

		OctreeNode(const Box& box, const long unsigned int ID, unsigned int (&ijk)[3], unsigned int depth, root_t* root):
				box(box), ID(ID), ijk{ijk[0], ijk[1], ijk[2]}, _isdivided(false), _depth(depth), _root(root) {}
		
		~OctreeNode(){
			if (_isdivided){
				for (int i=0; i<8; i++){
					delete _children[i];
				}
			}
		}

		void make_children();
		node_t* findleaf(const Point3& point); //find leaf node that contains the given point. Return NULL if there is none.
		node_t* findnode(unsigned int depth, const Point3& point); //find node that contains the given point at the given depth. Return NULL if there is none.
		
		inline bool isdivided() const {return _isdivided;}

		Box box;
		const long unsigned int ID;
		const unsigned int ijk[3];

		void get_global_vertex_index(long unsigned int (&global_index)[8]) const;
	
	protected:
		//node identifiers
		bool _isdivided;
		unsigned int _depth;
		
		//tree structure
		root_t* _root;
		node_t* _children[8];
	};


	////////////////////// NODE IMPLEMENTATIONS //////////////////////////////
	template<typename root_t, typename node_t>
	void OctreeNode<root_t, node_t>::make_children(){
		if (_isdivided){
			return;
		}

		//create children
		_isdivided = true;
		_root->_nleaves += 7;
		unsigned int c_ijk[3];

		c_ijk[0] = 2*ijk[0];   c_ijk[1] = 2*ijk[1];   c_ijk[2]= 2*ijk[2];
		_children[0] = new node_t(Box(box[0], box.center()), ID*8+1, c_ijk, _depth+1, _root);

		c_ijk[0] = 2*ijk[0]+1; c_ijk[1] = 2*ijk[1];   c_ijk[2]= 2*ijk[2];
		_children[1] = new node_t(Box(box[1], box.center()), ID*8+2, c_ijk, _depth+1, _root);

		c_ijk[0] = 2*ijk[0];   c_ijk[1] = 2*ijk[1]+1; c_ijk[2]= 2*ijk[2];
		_children[2] = new node_t(Box(box[2], box.center()), ID*8+3, c_ijk, _depth+1, _root);

		c_ijk[0] = 2*ijk[0]+1; c_ijk[1] = 2*ijk[1]+1; c_ijk[2]= 2*ijk[2];
		_children[3] = new node_t(Box(box[3], box.center()), ID*8+4, c_ijk, _depth+1, _root);

		c_ijk[0] = 2*ijk[0];   c_ijk[1] = 2*ijk[1];   c_ijk[2]= 2*ijk[2]+1;
		_children[4] = new node_t(Box(box[4], box.center()), ID*8+5, c_ijk, _depth+1, _root);

		c_ijk[0] = 2*ijk[0]+1; c_ijk[1] = 2*ijk[1];   c_ijk[2]= 2*ijk[2]+1;
		_children[5] = new node_t(Box(box[5], box.center()), ID*8+6, c_ijk, _depth+1, _root);

		c_ijk[0] = 2*ijk[0];   c_ijk[1] = 2*ijk[1]+1; c_ijk[2]= 2*ijk[2]+1;
		_children[6] = new node_t(Box(box[6], box.center()), ID*8+7, c_ijk, _depth+1, _root);

		c_ijk[0] = 2*ijk[0]+1; c_ijk[1] = 2*ijk[1]+1; c_ijk[2]= 2*ijk[2]+1;
		_children[7] = new node_t(Box(box[7], box.center()), ID*8+8, c_ijk, _depth+1, _root);
	}


	template<typename root_t, typename node_t>
	node_t* OctreeNode<root_t, node_t>::findleaf(const Point3& point){
		if (_isdivided){
			for (int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->box.contains(point)){
					return _children[c_idx]->findleaf(point);
				}
			}
		}

		if (box.contains(point)){
			return reinterpret_cast<node_t*>(this);
		}

		return NULL;
	}


	template<typename root_t, typename node_t>
	node_t* OctreeNode<root_t, node_t>::findnode(const unsigned int depth, const Point3& point){
		if (box.contains(point) && depth >= _depth){
			if (depth == _depth){
				return reinterpret_cast<node_t*>(this);
			}
			if (_isdivided){
				for (int c_idx=0; c_idx<8; c_idx++){
					if (_children[c_idx]->_box.contains(point)){
						return _children[c_idx]->findnode(depth, point);
					}
				}
			}
		}

		return NULL;
	}



	template<typename root_t, typename node_t>
	void OctreeNode<root_t, node_t>::get_global_vertex_index(long unsigned int (&global_index)[8]) const{
		//compute global index
		long unsigned int M2 = pow(2,_root->_maxdepth) + 1; //maximum number of vertices in side length
		long unsigned int Mm2 = pow(2, _root->_maxdepth-_depth); //conversion factor from current depth to _maxdepth

		global_index[0] = Mm2*(ijk[0]   + M2*(ijk[1]   + M2*(ijk[2]  )));
		global_index[1] = Mm2*(ijk[0]+1 + M2*(ijk[1]   + M2*(ijk[2]  )));
		global_index[2] = Mm2*(ijk[0]   + M2*(ijk[1]+1 + M2*(ijk[2]  )));
		global_index[3] = Mm2*(ijk[0]+1 + M2*(ijk[1]+1 + M2*(ijk[2]  )));
		global_index[4] = Mm2*(ijk[0]   + M2*(ijk[1]   + M2*(ijk[2]+1)));
		global_index[5] = Mm2*(ijk[0]+1 + M2*(ijk[1]   + M2*(ijk[2]+1)));
		global_index[6] = Mm2*(ijk[0]   + M2*(ijk[1]+1 + M2*(ijk[2]+1)));
		global_index[7] = Mm2*(ijk[0]+1 + M2*(ijk[1]+1 + M2*(ijk[2]+1)));
	}
}





#endif