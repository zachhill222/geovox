#ifndef SIMPLEX_H
#define SIMPLEX_H

#include "util/point.hpp"
#include <iostream>

namespace GeoVox::util{
	class Simplex{
	public:
		Simplex(){
			_vertex[0] = Point(0,0,0);
			_vertex[1] = Point(1,0,0);
			_vertex[2] = Point(0,1,0);
			_vertex[3] = Point(0,0,1);
		}

		Simplex(const Point& p1, const Point& p2, const Point& p3, const Point& p4){
			_vertex[0] = p1;
			_vertex[1] = p2;
			_vertex[2] = p3;
			_vertex[3] = p4;
		}
		
		Point operator[](int idx) const;
		Point& operator[](int idx);

		void print(std::ostream &stream) const;

		bool contains(const Point& point) const;

	private:
		Point _vertex[4];
	};
}




#endif