#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

#include "geometry/particles.hpp"
#include "geometry/collisions.hpp"

#include "mesh/vtk_linear_elements.hpp"
#include "mesh/vtk_convex_cell.hpp"
#include "mesh/mesh.hpp"
#include "mesh/nodes.hpp"

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
// using Vertex = GeoVox::mesh::Vertex;

namespace GeoVox::geometry{
	class AssemblyNode;
	class Assembly;
	using OctreeNode = GeoVox::util::OctreeNode<Assembly, AssemblyNode, SuperEllipsoid>;

	class AssemblyNode : public OctreeNode{
	public:
		AssemblyNode() : OctreeNode() {}

		AssemblyNode(const Box& box, const long unsigned int ID, unsigned int (&ijk)[3], unsigned int depth, Assembly* root) : 
			OctreeNode(box, ID, ijk, depth, root) {}
		
		// void make_children();
		// void new_make_children();

		// bool is_gradiated();
		bool in_particle(const Point3& point) const;
		bool data_valid(const SuperEllipsoid& P) const override;
		void push_data_to_children() override;

		int nvert() const;
		int nvert(long unsigned int d_idx) const;

		void get_global_vertex_index(long unsigned int (&global_index)[8]) const;
		void divide(const int n_divisions);
		void divide(); //divide to balance max_data_per_leaf
		// int _nvert; //maximum number of vertices contained by a SINGLE particle (leaves only)

		// void insert_particle(const SuperEllipsoid P);
		// std::vector<SuperEllipsoid> local_particles;

	protected:
		// std::vector<long unsigned int> _particle_index;
		// int _nvert;
		void makeElements(const std::map<long unsigned int, long unsigned int>& reduced_index, std::vector<std::vector<long unsigned int>> &elem2node, std::vector<int> &elemMarkers) const;
		void create_point_global_index_maps(std::vector<Point3>& points, std::map<long unsigned int, long unsigned int>& reduced_index) const;
	};


	class Assembly : public AssemblyNode {
	public:
		Assembly() : AssemblyNode(), _nleaves(1), _maxdepth(0), max_data_per_leaf(8) {
			_root = this;
		}

		Assembly(const std::string particle_file) : AssemblyNode(), _nleaves(1), _maxdepth(0), max_data_per_leaf(8) {
			_root = this;
			readfile(particle_file);
			for (long unsigned int i=0; i<_particles.size(); i++){
				_data.push_back(_particles[i]);
			}
		}

		void gradiate(); //ensure depth changes by at most one between neighbors

		void readfile(const std::string fullfile);
		void print(std::ostream &stream) const;

		Mesh make_voxel_mesh() const; //turn octree into a voxel mesh
		Mesh make_mixed_mesh() const; //turn octree into a mesh with voxels and hexahedrons. Points near the boundary of a particle are moved to the boundary.

		void save_geometry(const std::string filename, const Box& box, const int N[3]) const;
		void save_geometry(const std::string filename, const int N[3]) const;

		std::vector<SuperEllipsoid> _particles;

		void _setbbox();

		long unsigned int _nleaves;
		unsigned int _maxdepth;
		const long unsigned int max_data_per_leaf;
	};



}

#endif