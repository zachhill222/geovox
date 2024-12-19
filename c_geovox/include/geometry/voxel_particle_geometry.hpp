#ifndef VOXEL_PARTICLE_GEOMETRY_H
#define VOXEL_PARTICLE_GEOMETRY_H

#include "constants.hpp"
#include "geometry/assembly.hpp"
#include "mesh/vtk_structured.hpp"
#include "util/box.hpp"

#include <vector>
#include <array>
#include <set>
#include <string>
#include <iostream>

#include <omp.h>

using Assembly = GeoVox::geometry::Assembly;
using StructuredPoints = GeoVox::mesh::StructuredPoints;
using Box = GeoVox::util::Box;

namespace GeoVox::geometry{
	class VoxelParticleGeometry : public StructuredPoints {
	public:
		VoxelParticleGeometry() : StructuredPoints() {};
		VoxelParticleGeometry(const Assembly &Assem, const Box box, long unsigned int N[3]) : StructuredPoints(box, N), A(&Assem) {initialize();};
		VoxelParticleGeometry(const Assembly* Assem, const Box box, long unsigned int N[3]) : StructuredPoints(box, N), A(Assem) {initialize();};
		VoxelParticleGeometry(const Assembly &Assem, long unsigned int N[3]) : StructuredPoints(Assem.box, N), A(&Assem) {initialize();};
		VoxelParticleGeometry(const Assembly* Assem, long unsigned int N[3]) : StructuredPoints(Assem->box, N), A(Assem) {initialize();};

		
		//boundary conditions
		bool wallBC[6] {0}; //xlow, xhigh, ylow, yhigh, zlow, zhigh

		//separate the void space into disjoint (orthogonal connectivity) regions.
		//regions connected to inlet/outlet boundaries are labeled with poitive integers.
		//regions not connected to an inlet/outlet boundaries are labeled with negative integers.
		void compute_connectivity();

		void initialize(); //mark solid phase as SOLID_MARKER and set all others to UNDEFINED_MARKER

		void print(std::ostream &stream) const;
	private:
		Assembly const* A;

		bool find_unmarked_face(const int n, long unsigned int &ii, long unsigned int &jj, long unsigned int &kk);
		long unsigned int spread(std::vector<long unsigned int> &active_index);

	};






}
#endif