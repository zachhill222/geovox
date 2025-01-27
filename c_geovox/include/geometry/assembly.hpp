#ifndef ASSEMBLY_H
#define ASSEMBLY_H



#include "constants.hpp"

#include "util/point.hpp"
#include "util/box.hpp"
#include "util/octree.hpp"

#include "geometry/particles.hpp"
#include "geometry/collisions.hpp"
#include "geometry/voxel_particle_geometry.hpp"

#include <vector>
#include <map>
#include <initializer_list>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>


using SuperEllipsoid = GeoVox::geometry::SuperEllipsoid;
using Box = GeoVox::util::Box;

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

		// COLUMN OPTIONS:
		// -id (IDENTIFIER, int)
		// -rrr (TRIPLE RADIUS, double[3])
		// -xyz (CENTER, double[3])
		// -eps (SHAPE PARAMETERS, double[2])
		// -v (VOLUME, double)
		// -q (QUATERNION, double[4])
		// -l (BOUNDING BOX LENGTH, double[3])
		Assembly(const std::string particle_file, const std::string columns) : AssemblyNode(), _nleaves(1), _maxdepth(0), max_data_per_leaf(8) {
			_root = this;
			readfile(particle_file, columns);
			for (long unsigned int i=0; i<_particles.size(); i++){
				_data.push_back(_particles[i]);
			}
		}

		void gradiate(); //ensure depth changes by at most one between neighbors

		// void readfile(const std::string fullfile);
		void readfile(const std::string fullfile, const std::string columns);
		void print(std::ostream &stream) const;
		std::string tostr() const;

		VoxelParticleGeometry make_structured_mesh(const Box& subbox, const long unsigned int N[3]) const;
		inline VoxelParticleGeometry make_structured_mesh(const long unsigned int N[3]) const {return make_structured_mesh(box, N);}

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