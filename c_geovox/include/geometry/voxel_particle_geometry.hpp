#ifndef VOXEL_PARTICLE_GEOMETRY_H
#define VOXEL_PARTICLE_GEOMETRY_H

#include "constants.hpp"
#include "mesh/vtk_structured.hpp"
#include "util/box.hpp"

#include <vector>
#include <array>
#include <set>
#include <string>
#include <sstream>
#include <iostream>

#include <omp.h>

using StructuredPoints = GeoVox::mesh::StructuredPoints;
using Box = GeoVox::util::Box;

namespace GeoVox::geometry{
	class VoxelParticleGeometry : public StructuredPoints {
	public:
		VoxelParticleGeometry() : StructuredPoints() {};
		VoxelParticleGeometry(const Box& box, const long unsigned int N[3]) : StructuredPoints(box, N, 3) {};

		
		//boundary conditions// 1 for wall bc (homogeneous Dirichlet)
		bool wall_bc[6] {0}; //xlow, xhigh, ylow, yhigh, zlow, zhigh

		//separate the void space into disjoint (orthogonal connectivity) regions.
		//regions connected to inlet/outlet boundaries are labeled with poitive integers.
		//regions not connected to an inlet/outlet boundaries are labeled with negative integers.
		void compute_connectivity();

		void initialize(); //mark solid phase as SOLID_MARKER and set all others to UNDEFINED_MARKER

		void print(std::ostream &stream) const;
		std::string tostr() const;
	private:
		bool find_unmarked_boundary_voxel(std::vector<long unsigned int> &active_index, const long unsigned int max_voxels=16) const;
		bool find_unmarked_voxel(std::vector<long unsigned int> &active_index, const long unsigned int max_voxels=16) const ;
		long unsigned int spread(std::vector<long unsigned int> &active_index);

		void merge_regions(const int mkr_low, const int mkr_high);
	};






}
#endif