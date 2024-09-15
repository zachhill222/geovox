#ifndef POLYTOPES_H
#define POLYTOPES_H

#include "util/point.hpp"
#include <iostream>
#include <vector>

using Point = GeoVox::util::Point;
namespace GeoVox::geometry{
	class Polytope{
	public:
		Polytope(const int npts){
			_points.reserve(npts);
		}

		Point operator[](int idx) const;
		Point& operator[](int idx);

		int len() const;

		void addpoint(const Point& point);

		Point support(const Point& direction) const;

	protected:
		std::vector<Point> _points;
	};



	class Simplex : public Polytope {
	public:
		Simplex(): Polytope(4) {
			_points[0] = Point(0,0,0);
			_points[1] = Point(1,0,0);
			_points[2] = Point(0,1,0);
			_points[3] = Point(0,0,1);
		}

		Simplex(const Point& p1, const Point& p2, const Point& p3, const Point& p4): Polytope(4) {
			_points[0] = p1;
			_points[1] = p2;
			_points[2] = p3;
			_points[3] = p4;
		}

		void print(std::ostream &stream) const;

		bool contains(const Point& point) const;
	};
}


#endif