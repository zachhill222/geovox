#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include "util/quaternion.hpp"
#include "util/point.hpp"
#include "util/box.hpp"

#include "geometry/particles.hpp"
#include "geometry/collisions.hpp"

#include <vector>
#include <initializer_list>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>



using SuperEllipsoid = GeoVox::geometry::SuperEllipsoid;
using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;

namespace GeoVox::geometry{
	class Node{
	public:
		Node(const int depth=0) : _root(NULL), _parent(NULL), _depth(depth), _ID(0), _box(Box(Point3(0,0,0), Point3(1,1,1))), _isdivided(false) {}
		Node(const Point3& low, const Point3& high, const int depth=0) : _root(NULL), _parent(NULL), _depth(depth), _ID(0), _box(Box(low, high)), _isdivided(false) {}

		~Node(){
			if (_isdivided){
				for (int i=0; i<8; i++){
					delete _children[i];
				}
			}
		}

		Node(const Node& other){
			_root           = other._root;
			_parent         = other._parent;
			_maxdepth       = other._maxdepth;
			_depth          = other._depth;
			_ID             = other._ID;
			_particles      = other._particles;
			_particle_index = other._particle_index;
			_box            = other._box;
			_isdivided      = other._isdivided;

			_children[0]    = other._children[0];
			_children[1]    = other._children[1];
			_children[2]    = other._children[2];
			_children[3]    = other._children[3];
			_children[4]    = other._children[4];
			_children[5]    = other._children[5];
			_children[6]    = other._children[6];
			_children[7]    = other._children[7];
		}

		Node operator[](long unsigned int idx) const; //ID idx relative to current node. result._ID == idx if called from root node.
		void insert_particle(const long unsigned int idx, const SuperEllipsoid& P);




		Node* _root;
		Node* _parent;
		int _maxdepth;
		int _depth;
		long unsigned int _ID; //PARENT_ID*8 + CHILD_ID

		std::vector<SuperEllipsoid> _particles; //only for _root node
		std::vector<long unsigned int> _particle_index;
		Box _box;
		
		bool _isdivided;
		Node* _children[8];
	};


	class Assembly : protected Node {
	public:
		Assembly() : Node() {};
		
		Assembly(const int size) : Node(0) {
			_particles.reserve(size);
		}

		Assembly(const std::string particle_file) : Node(0) {
			readfile(particle_file);
			_setbbox();
		}

		SuperEllipsoid operator[](int idx) const;
		SuperEllipsoid& operator[](int idx);
		void readfile(const std::string fullfile);
		void print(std::ostream &stream) const;

	private:
		void _setbbox();
	};



}

#endif