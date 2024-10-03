#ifndef VTK_STRUCTURED_H
#define VTK_STRUCTURED_H

#include "util/box.hpp"
#include "util/point.hpp"


#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using Point3 = GeoVox::util::Point<3>;
using Box = GeoVox::util::Box;




namespace GeoVox::mesh{
	class StructuredPoints{
	public:
		StructuredPoints(const Box& box, const long unsigned int N[3]) :  box(box), N{N[0], N[1], N[2]}, H(((box.high()-box.low())/Point3(N[0], N[1], N[2]))) {}
		StructuredPoints(const Point3& low, const Point3& high, const long unsigned int N[3]) :  box(Box(low, high)), N{N[0], N[1], N[2]}, H(((box.high()-box.low())/Point3(N[0], N[1], N[2]))) {}
		StructuredPoints(const Box& box, const std::string geofile) : box(box) {readfile(geofile);};

		//mesh information
		std::vector<bool> pointMarkers;
		Box box;
		inline Point3 spacing() const {return H;}
		inline Box bounds() const {return box;}

		//access methods
		inline Point3 vertex(long unsigned int i, long unsigned int j, long unsigned int k) const {return box.low() + H*Point3(i,j,k);}
		inline long unsigned int index(long unsigned int i, long unsigned int j, long unsigned int k) {return i + N[0]*( j + N[1]*k);}

		//fileio
		void saveas(const std::string filename) const;
		void readfile(const std::string filename);
	private:
		//mesh parameters
		long unsigned int N[3];
		Point3 H;
	};
	
	
}

#endif