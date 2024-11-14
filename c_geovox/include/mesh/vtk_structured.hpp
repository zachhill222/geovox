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

#include <omp.h>

using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;




namespace GeoVox::mesh{
	class StructuredPoints{
	public:
		StructuredPoints () {}
		StructuredPoints(const Box& box, const long unsigned int N[3]) :  box(box), N{N[0], N[1], N[2]}, H((box.high()-box.low()).array()/Point3(N[0]-1, N[1]-1, N[2]-1).array()) {markers = std::vector<int>(N[0]*N[1]*N[2], 0);}
		StructuredPoints(const Point3& low, const Point3& high, const long unsigned int N[3]) :  box(Box(low, high)), N{N[0], N[1], N[2]}, H((box.high()-box.low()).array()/Point3(N[0]-1, N[1]-1, N[2]-1).array()) {markers = std::vector<int>(N[0]*N[1]*N[2], 0);}
		StructuredPoints(const Box& box, const std::string geofile) : box(box) {readfile(geofile);};
		
		//access methods
		inline Point3 idx2point(long unsigned int i, long unsigned int j, long unsigned int k) const {return Point3(box.low()[0]+H[0]*i, box.low()[1]+H[1]*j, box.low()[2]+H[2]*k);}
		inline long unsigned int index(long unsigned int i, long unsigned int j, long unsigned int k) const {return i + N[0]*( j + N[1]*k);}
		bool index2ijk(long unsigned int l, long unsigned int &i, long unsigned int &j, long unsigned int &k) const;
		inline int operator()(long unsigned int i, long unsigned int j, long unsigned int k) const {return markers[index(i,j,k)];}

		//fileio
		void saveas(const std::string filename) const;
		void readfile(const std::string filename);
		
		//mesh information
		std::vector<int> markers;
		Box box;
		Eigen::Array<long unsigned int, 1, 3> N;
		Point3 H;

		//set all markers
		void set_all_markers(const int mkr);
		void replace_marker(const int old_mkr, const int new_mkr);
	};
	
	
}

#endif