#include "mesh/nodes.hpp"

namespace GeoVox::mesh{
	void MeshNodeNode::make_children(){
		if (_isdivided){
			return;
		}

		OctreeNode::make_children();

		// std::cout << "initialized children\n";

		for (long unsigned int p_idx=0; p_idx<_points.size(); p_idx++){
			for (int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->box.contains(_points[p_idx])){
					_children[c_idx]->_points.push_back(_points[p_idx]);
					break;
				}
			}
		}

		//set number of nodes
		for (int c_idx=0; c_idx<8; c_idx++){
			_children[c_idx]->_npoints = _children[c_idx]->_points.size();
		}

		//only leaf nodes contain points.
		_points.clear();
	}

	// long unsigned int MeshNodeNode::npoints() const{
	// 	if (_isdivided){
	// 		long unsigned int result = 0;
	// 		for (int c_idx=0; c_idx<8; c_idx++){
	// 			result += _children[c_idx]->npoints();
	// 		}
	// 		return result;
	// 	}
	// 	return _points.size();
	// }

	void MeshNodeNode::insert_point(const Point3 newpoint){
		// std::cout << "insert_point\n";
		if (_isdivided){
			// std::cout << "\tdivided\n";
			_npoints += 1;

			for (int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->box.contains(newpoint)){
					_children[c_idx]->insert_point(newpoint); //biased towards child 0
					return;
				}
			}
		}else{
			// std::cout << "\tnot divided: " << _points.size() << std::endl;
			// std::cout << _root->max_points << std::endl;


			if (_points.size() >= _root->max_points){
				//need to subdivide
				// std::cout << "make _children\n";
				make_children();
				insert_point(newpoint);

			}else{
				// std::cout << "adding point\n";
				_points.push_back(newpoint);
				_npoints += 1;
			}
		}
	}

	void MeshNodeNode::insert_point(std::vector<Point3> newpoints){
		if (newpoints.size() == 0){
			return;
		}


		if (_isdivided){
			_npoints += newpoints.size();

			//separate newpoints into 8 sets
			std::vector<Point3> split_points[8];
			for (int c_idx=0; c_idx<8; c_idx++){
				split_points[c_idx].reserve(newpoints.size());
			}

			for (long unsigned int p_idx=0; p_idx<newpoints.size(); p_idx++){
				for (int c_idx=0; c_idx<8; c_idx++){
					if (_children[c_idx]->box.contains(newpoints[p_idx])){
						split_points[c_idx].push_back(newpoints[p_idx]);
						break;
					}
				}
			}

			//manage memory
			newpoints.clear();
			for (int c_idx=0; c_idx<8; c_idx++){
				split_points[c_idx].shrink_to_fit();
			}

			//push points to children
			for (int c_idx=0; c_idx<8; c_idx++){
				_children[c_idx]->insert_point(split_points[c_idx]);
			}
		} else if (newpoints.size() + _points.size() > _root->max_points){
			make_children();
			insert_point(newpoints); //call on self to go into top branch
		} else{
			for (long unsigned int p_idx=0; p_idx<newpoints.size(); p_idx++){
				_points.push_back(newpoints[p_idx]);
			}
			_points.shrink_to_fit();
			_npoints += newpoints.size();
		}
	}

	Point3 MeshNodeNode::operator[](const long unsigned int idx) const{
		if (_isdivided) {
			long unsigned int pt_counter = 0;
			for (int c_idx=0; c_idx<8; c_idx++){
				if (pt_counter + _children[c_idx]->npoints() > idx){
					return _children[c_idx]->operator[](idx-pt_counter);
				}else{
					pt_counter += _children[c_idx]->npoints();
				}
			}
		}
		if (idx >= _points.size()){
			std::cout << idx << std::endl;
			std::invalid_argument("ERROR");
		}

		return _points[idx];
	}


	long unsigned int MeshNodeNode::find(const Point3& point) const{
		if (_isdivided){
			long unsigned int result = 0;
			for (int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->box.contains(point)){
					return result + _children[c_idx]->find(point);
				}else{
					result += _children[c_idx]->npoints();
				}
			}
		} else {
			for (long unsigned int p_idx=0; p_idx<_points.size(); p_idx++){
				if (_points[p_idx]==point){
					return p_idx;
				}
			}
		}
	std::invalid_argument("Couldn't find point.");
	return 0;
	}
}

