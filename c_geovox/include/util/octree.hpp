#ifndef OCTREE_H
#define OCTREE_H

#define MAX_OCTREE_DEPTH 16


#include "util/point.hpp"
#include "util/box.hpp"

#include <vector>
#include <stdexcept>

using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;

namespace GeoVox::util{

	template<typename root_t, typename node_t, typename data_t>
	class OctreeNode{
	public:
		OctreeNode() : box(Box()), ID(0), ijk{0,0,0}, _isdivided(false), _depth(0), _root(NULL) {}

		OctreeNode(const Box& box) : box(box), ID(0), ijk{0,0,0}, _isdivided(false), _depth(0), _root(NULL) {}

		OctreeNode(const Box& box, const long unsigned int ID, unsigned int (&ijk)[3], unsigned int depth, root_t* root):
				box(box), ID(ID), ijk{ijk[0], ijk[1], ijk[2]}, _isdivided(false), _depth(depth), _root(root) {}
		
		virtual ~OctreeNode(){
			if (_isdivided){
				for (int i=0; i<8; i++){
					delete _children[i];
				}
			}
		}

		//data access
		// long unsigned int size() const;
		inline long unsigned int size() const {return _ndata;} //return number of stored data in this node and below
		void set_ndata();
		data_t operator[](long unsigned int idx) const;
		inline unsigned int depth() const {return _depth;}

		inline bool isdivided() const {return _isdivided;}
		inline node_t const* child_const(int c_idx) const {return reinterpret_cast<node_t const*>(_children[c_idx]);}


		long unsigned int find(const data_t& value) const; //return index of first instance of stored value. Raises an error if value cannot be found.

		//data storage logic
		virtual void insert_data(data_t value);
		virtual bool data_valid(const data_t& value) const; //IMPLEMENTED IN INHERITED CLASS
		virtual void push_data_to_children();
		void make_children();
		
		//access nodes
		node_t* findleaf(const Point3& point); //find leaf node that contains the given point. Return NULL if there is none.
		node_t* findnode(unsigned int depth, const Point3& point); //find node that contains the given point at the given depth. Return NULL if there is none.
		node_t const* findleaf_const(const Point3& point) const; //same as findnode, can be used in const member functions.

		Box box; //TODO: this should be const, but have trouble in _root
		const long unsigned int ID;
		const unsigned int ijk[3];


		//data
		std::vector<data_t> _data;
	protected:
		long unsigned int _ndata;

		//node identifiers
		bool _isdivided;
		unsigned int _depth;
		
		//tree structure
		root_t* _root;
		node_t* _children[8];
	};


	////////////////////// NODE IMPLEMENTATIONS //////////////////////////////
	template<typename root_t, typename node_t, typename data_t>
	bool OctreeNode<root_t, node_t, data_t>::data_valid(const data_t& value) const{
		throw std::invalid_argument("This function should never be called. Each octree class must implement a method: bool data_valid(const data_t& value) const override");
		return false;
	}


	// template<typename root_t, typename node_t, typename data_t>
	// long unsigned int OctreeNode<root_t, node_t, data_t>::size() const{
	// 	if (_isdivided){
	// 		long unsigned int result = 0;
	// 		for (int c_idx=0; c_idx<8; c_idx++){
	// 			result += _children[c_idx]->size();
	// 		}
	// 		return result;
	// 	}

	// 	return _data.size();
	// }


	template<typename root_t, typename node_t, typename data_t>
	void OctreeNode<root_t, node_t, data_t>::set_ndata(){
		if (_isdivided){
			_ndata = 0;
			for (int c_idx=0; c_idx<8; c_idx++){
				_ndata += _children[c_idx]->size();
			}
			return;
		}

		_ndata = _data.size();
	}


	template<typename root_t, typename node_t, typename data_t>
	void OctreeNode<root_t, node_t, data_t>::make_children(){
		// std::cout << "make_children(" << ID << "): _isdivided= " << _isdivided << " depth= " << _depth << std::endl;
		if (_isdivided){
			return;
		}

		if (_depth >= MAX_OCTREE_DEPTH){
			throw std::runtime_error("MAX_OCTREE_DEPTH exceeded. Re-define MAX_OCTREE_DEPTH if this depth is allowed.");
		}


		//create children
		_isdivided = true;
		_root->_nleaves += 7;
		_root->_maxdepth = std::max(_root->_maxdepth, _depth+1);

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

		push_data_to_children();

		//set number of nodes
		for (int c_idx=0; c_idx<8; c_idx++){
			_children[c_idx]->set_ndata();
		}

		//only leaf nodes contain data
		_data.clear();
	}



	template<typename root_t, typename node_t, typename data_t>
	void OctreeNode<root_t, node_t, data_t>::push_data_to_children(){
		//push data to children and clear current data
		for (long unsigned int d_idx=0; d_idx<_data.size(); d_idx++){
			for (int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->data_valid(_data[d_idx])){
					_children[c_idx]->_data.push_back(_data[d_idx]);
					break;
				}
			}
		}
	}


	template<typename root_t, typename node_t, typename data_t>
	void OctreeNode<root_t, node_t, data_t>::insert_data(data_t value){
		if (_isdivided){
			_ndata+=1;
			for (int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->data_valid(value)){
					_children[c_idx]->insert_data(value);
					return;
				}
			}
			return;
		}

		_data.push_back(value);
		_ndata += 1;
		if (_data.size()>_root->max_data_per_leaf){
			make_children();
		}

		return;
	}


	template<typename root_t, typename node_t, typename data_t>
	data_t OctreeNode<root_t, node_t, data_t>::operator[](const long unsigned int idx) const{
		if (_isdivided) {
			long unsigned int pt_counter = 0;
			for (int c_idx=0; c_idx<8; c_idx++){
				if (pt_counter + _children[c_idx]->size() > idx){
					return _children[c_idx]->operator[](idx-pt_counter);
				}else{
					pt_counter += _children[c_idx]->size();
				}
			}
		}
		return _data[idx];
	}


	template<typename root_t, typename node_t, typename data_t>
	long unsigned int OctreeNode<root_t, node_t, data_t>::find(const data_t& value) const{
		if (_isdivided){
			long unsigned int result = 0;
			for (int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->data_valid(value)){
					return result + _children[c_idx]->find(value);
				}else{
					result += _children[c_idx]->size();
				}
			}
		}else{
			for (long unsigned int d_idx=0; d_idx<_data.size(); d_idx++){
				if (_data[d_idx] == value){
					return d_idx;
				}
			}
		}

		throw std::invalid_argument("Couldn't find specified value.");
		return 0;
	}


	template<typename root_t, typename node_t, typename data_t>
	node_t* OctreeNode<root_t, node_t, data_t>::findleaf(const Point3& point){
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


	template<typename root_t, typename node_t, typename data_t>
	node_t const* OctreeNode<root_t, node_t, data_t>::findleaf_const(const Point3& point) const{
		if (_isdivided){
			for (int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->box.contains(point)){
					return _children[c_idx]->findleaf_const(point);
				}
			}
		}

		if (box.contains(point)){
			return reinterpret_cast<node_t const*>(this);
		}

		return NULL;
	}


	template<typename root_t, typename node_t, typename data_t>
	node_t* OctreeNode<root_t, node_t, data_t>::findnode(const unsigned int depth, const Point3& point){
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



}





#endif