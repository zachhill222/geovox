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
		Node(const int depth=0) : _root(NULL), _parent(NULL), _depth(depth), _box(Box(Point3(0,0,0), Point3(1,1,1))), _isdivided(false) {}
		Node(const Point3& low, const Point3& high, const int depth=0) : _root(NULL), _parent(NULL), _depth(depth), _box(Box(low, high)), _isdivided(false) {}

		~Node(){
			if (_isdivided){
				for (int i=0; i<8; i++){
					delete _children[i];
				}
			}
		}

		void insert_particle(const SuperEllipsoid& P);
		Node* _root;
		Node* _parent;
		int _maxdepth;
		const int _depth;

	protected:
		std::vector<SuperEllipsoid> _particles;
		Box _box;
		
	private:
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