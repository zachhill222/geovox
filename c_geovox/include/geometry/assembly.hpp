#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include "util/quaternion.hpp"
#include "util/point.hpp"
#include "util/box.hpp"

#include "geometry/particles.hpp"
#include "geometry/collisions.hpp"

#include "mesh/mesh.hpp"

#include <vector>
#include <map>
#include <initializer_list>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>



using SuperEllipsoid = GeoVox::geometry::SuperEllipsoid;
using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;
using Mesh = GeoVox::mesh::Mesh;

namespace GeoVox::geometry{
	class Node;
	class Root;
	class Assembly;


	class Node{
	public:
		Node(const int depth) : _root(NULL), _parent(NULL), _depth(depth), _ID(0), _nvert(0), _isdivided(false) {}
		
		Node(const Point3& low, const Point3& high): _root(NULL), _parent(NULL), _nvert(0), _box(Box(low, high)), _isdivided(false) {}

		~Node(){
			if (_isdivided){
				for (int i=0; i<8; i++){
					delete _children[i];
				}
			}
		}


		// void insert_particle(const long unsigned int idx, const SuperEllipsoid& P);

		Root* _root;
		Node* _parent;
		unsigned int _depth;
		
		long unsigned int _ID; //PARENT_ID*8 + CHILD_ID + 1
		unsigned int _ijk[3]; //ijk indices of current voxel at _depth
		
		int _nvert; //maximum number of vertices contained by a SINGLE particle (leaves only)

		std::vector<long unsigned int> _particle_index;
		Box _box;
		
		bool _isdivided;
		Node* _children[8];

		void divide();
		bool in_particle(const Point3& point) const;
		void move_to_particle_surface(Point3& point) const;

		void makeElements(const std::map<long unsigned int, long unsigned int>& reduced_index, std::vector<std::vector<long unsigned int>> &elem2node, std::vector<int> &elemMarkers) const;
		void create_point_global_index_maps(std::vector<Point3>& point_map, std::map<long unsigned int, long unsigned int>& reduced_index) const;
		void get_global_vertex_index(long unsigned int (&global_index)[8]) const;
	};


	class Root : public Node {
	public:
		Root(const int maxdepth=0, const Box box=Box(Point3(0,0,0), Point3(1,1,1))) : Node(0), _maxdepth(maxdepth) {
			_ijk[0] = 0;
			_ijk[1] = 0;
			_ijk[2] = 0;

			_nleaves = 1;
			_root    = this;
		}


		unsigned int _maxdepth;
		long unsigned int _nleaves;
		std::vector<SuperEllipsoid> _particles;
	};


	class Assembly : public Root {
	public:
		Assembly() : Root(0) {};

		Assembly(const std::string particle_file) : Root(0) {
			readfile(particle_file);
		}

		SuperEllipsoid operator[](int idx) const;
		SuperEllipsoid& operator[](int idx);
		void readfile(const std::string fullfile);
		void print(std::ostream &stream) const;
		// void print_tree(std::ostream &stream) const;
		void make_tree(int maxdepth);

		Mesh make_voxel_mesh() const; //turn octree into a voxel mesh
		Mesh make_mixed_mesh() const; //turn octree into a mesh with voxels and hexahedrons. Points near the boundary of a particle are moved to the boundary.

		void save_geometry(const std::string filename, const Box& box, const int N[3]) const;
		void save_geometry(const std::string filename, const int N[3]) const;


	private:
		void _setbbox();
	};



}

#endif