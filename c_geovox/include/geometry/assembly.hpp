#ifndef ASSEMBLY_H
#define ASSEMBLY_H

#define MIN_ASSEMBLY_NUMBER_OF_PARTICLES 0


#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

#include "geometry/particles.hpp"
#include "geometry/collisions.hpp"

#include "mesh/vtk_linear_elements.hpp"
#include "mesh/vtk_convex_cell.hpp"
#include "mesh/mesh.hpp"
#include "mesh/vtk_structured.hpp"
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
using StructuredPoints = GeoVox::mesh::StructuredPoints;
// using Vertex = GeoVox::mesh::Vertex;

namespace GeoVox::geometry{
	class AssemblyNode;
	class Assembly;
	using OctreeNode = GeoVox::util::OctreeNode<Assembly, AssemblyNode, SuperEllipsoid>;

	class AssemblyNode : public OctreeNode{
	public:
		AssemblyNode() : OctreeNode(), _nvert(0) {}

		AssemblyNode(const Box& box, const long unsigned int ID, unsigned int (&ijk)[3], unsigned int depth, Assembly* root) : 
			OctreeNode(box, ID, ijk, depth, root), _nvert(0) {}

		// bool is_gradiated();
		bool in_particle(const Point3& point) const;
		bool data_valid(const SuperEllipsoid& P) const override;
		void push_data_to_children() override;

		void divide(const int n_divisions);
		void divide(); //divide to balance max_data_per_leaf

		int _nvert;
	protected:
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

		StructuredPoints make_structured_mesh(const Box& subbox, const long unsigned int N[3]) const;
		StructuredPoints make_structured_mesh(const long unsigned int N[3]) const;

		//FOR HYBGE ONLY. USE StructuredPoints
		void save_geometry(const std::string filename, const Box& box, const long unsigned int N[3]) const;
		inline void save_geometry(const std::string filename, const long unsigned int  N[3]) const{save_geometry(filename, box, N);}

		std::vector<SuperEllipsoid> _particles;

		void _setbbox();

		long unsigned int _nleaves;
		unsigned int _maxdepth;
		const long unsigned int max_data_per_leaf;
	};



}

#endif