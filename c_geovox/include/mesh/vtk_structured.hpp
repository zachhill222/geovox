#ifndef VTK_STRUCTURED_H
#define VTK_STRUCTURED_H

#include "constants.hpp"
#include "util/box.hpp"
#include "util/point.hpp"

#include "Eigen/Core"

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>

#include <omp.h>

using Point3 = GeoVox::util::Point3;
using Box = GeoVox::util::Box;


namespace GeoVox::mesh{
	class StructuredPoints {
	public:
		StructuredPoints () {}
		StructuredPoints(const Box& box, const long unsigned int _N[3], const int dof_location=3) :  periodic_bc{1,1,1}, dof_location(dof_location), box(box){
			N[0] = _N[0];
			N[1] = _N[1];
			N[2] = _N[2];

			H = box.sidelength().array()/Point3(_N[0], _N[1], _N[2]).array();

			switch (dof_location){
			case 0:
				if (!periodic_bc[0]){
					N[0] += 1;
				}
			case 1:
				if (!periodic_bc[1]){
					N[1] += 1;
				}
			case 2:
				if (!periodic_bc[2]){
					N[2] += 1;
				}
			}
			markers = std::vector<int>(N[0]*N[1]*N[2], 0);
		}
		StructuredPoints(const Point3& low, const Point3& high, const long unsigned int _N[3], const int dof_location=3) : periodic_bc{1,1,1}, dof_location(dof_location) {
			N[0] = _N[0];
			N[1] = _N[1];
			N[2] = _N[2];

			box = Box(low, high);
			H = box.sidelength().array()/Point3(_N[0], _N[1], _N[2]).array();

			switch (dof_location){
			case 0:
				if (!periodic_bc[0]){
					N[0] += 1;
				}
			case 1:
				if (!periodic_bc[1]){
					N[1] += 1;
				}
			case 2:
				if (!periodic_bc[2]){
					N[2] += 1;
				}
			}

			markers = std::vector<int>(N[0]*N[1]*N[2], 0);
		}
		// StructuredPoints(const Box& box, const std::string geofile, const int dof_location=3) : periodic_bc{1,1,1}, dof_location(dof_location), box(box) {readfile(geofile);};
		
		//access methods
		Point3 idx2point(long unsigned int i, long unsigned int j, long unsigned int k) const;
		inline long unsigned int index(long unsigned int i, long unsigned int j, long unsigned int k) const {return i + N[0]*( j + N[1]*k);}
		bool index2ijk(long unsigned int l, long unsigned int &i, long unsigned int &j, long unsigned int &k) const;
		inline int operator()(long unsigned int i, long unsigned int j, long unsigned int k) const {return markers[index(i,j,k)];}

		//neighbors
		bool periodic_bc[3];
		long unsigned int east(long unsigned int i, long unsigned int j, long unsigned int k) const;
		long unsigned int west(long unsigned int i, long unsigned int j, long unsigned int k) const;
		long unsigned int north(long unsigned int i, long unsigned int j, long unsigned int k) const;
		long unsigned int south(long unsigned int i, long unsigned int j, long unsigned int k) const;
		long unsigned int top(long unsigned int i, long unsigned int j, long unsigned int k) const;
		long unsigned int bottom(long unsigned int i, long unsigned int j, long unsigned int k) const;

		//degree of freedom location
		// 0 -> cell faces perpendicular to x-axis
		// 1 -> cell faces perpendicular to y-axis
		// 2 -> cell faces perpendicular to z-axis
		// 3 -> cell centroids
		int dof_location;

		//degree of freedom data
		Eigen::VectorXd _data;

		//fileio
		void saveas(const std::string filename) const;
		// void readfile(const std::string filename);
		
		//mesh information
		std::vector<int> markers;
		Box box;
		Eigen::Array<long unsigned int, 1, 3> N;
		// long unsigned int N[3];
		Point3 H;

		//set all markers
		void set_all_markers(const int mkr);
		void replace_marker(const int old_mkr, const int new_mkr);

		//count markers
		long unsigned int count(const int mkr) const;
		void unique_markers(std::vector<int> &mkr, std::vector<long unsigned int> &mkr_count) const;
	};
	
	
}

#endif