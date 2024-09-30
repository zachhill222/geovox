#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

#include "geometry/particles.hpp"
#include "geometry/collisions.hpp"

#include "mesh/vtk_linear.hpp"
#include "mesh/vtk_convex_cell.hpp"
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
using Vertex = GeoVox::mesh::Vertex;

namespace GeoVox::geometry{
	class AssemblyNode;
	class Assembly;
	using OctreeNode = GeoVox::util::OctreeNode<Assembly, AssemblyNode>;

	class AssemblyNode : public OctreeNode{
	public:
		AssemblyNode() : OctreeNode(), _nvert(0) {}

		AssemblyNode(const Box& box, const long unsigned int ID, unsigned int (&ijk)[3], unsigned int depth, Assembly* root) : 
			OctreeNode(box, ID, ijk, depth, root), _nvert(0) {}
		
		void make_children();
		bool is_gradiated();
		bool in_particle(const Point3& point) const;

		void divide();
		int _nvert; //maximum number of vertices contained by a SINGLE particle (leaves only)
	protected:
		
		std::vector<long unsigned int> _particle_index;
		
		void makeElements(const std::map<long unsigned int, long unsigned int>& reduced_index, std::vector<std::vector<long unsigned int>> &elem2node, std::vector<int> &elemMarkers) const;
		void create_point_global_index_maps(std::vector<Vertex>& points, std::map<long unsigned int, long unsigned int>& reduced_index) const;
	};


	class Assembly : public AssemblyNode {
	public:
		Assembly() : AssemblyNode(), _maxdepth(5), _nleaves(1) {
			_root = this;
		}

		Assembly(const std::string particle_file) : AssemblyNode(), _maxdepth(5), _nleaves(1) {
			_root = this;
			readfile(particle_file);
		}

		unsigned int _maxdepth;
		long unsigned int _nleaves;

		void gradiate(); //ensure depth changes by at most one between neighbors

		void readfile(const std::string fullfile);
		void print(std::ostream &stream) const;
		// void print_tree(std::ostream &stream) const;
		void make_tree(int maxdepth);

		Mesh make_voxel_mesh() const; //turn octree into a voxel mesh
		Mesh make_mixed_mesh() const; //turn octree into a mesh with voxels and hexahedrons. Points near the boundary of a particle are moved to the boundary.

		void save_geometry(const std::string filename, const Box& box, const int N[3]) const;
		void save_geometry(const std::string filename, const int N[3]) const;

		std::vector<SuperEllipsoid> _particles;

		void _setbbox();
	};



}

#endif