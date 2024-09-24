#ifndef POLYTOPES_H
#define POLYTOPES_H

#include "util/point.hpp"
#include <iostream>
#include <vector>
#include <initializer_list>

namespace GeoVox::geometry{
	using Point3 = GeoVox::util::Point<3>;
	
	class Polytope{
	public:
		Polytope(const int npts){
			_points.reserve(npts);
		}

		Polytope(std::initializer_list<Point3> list) : _points(list) {}

		Point3 operator[](int idx) const;
		Point3& operator[](int idx);

		int len() const;

		void addpoint(const Point3& point);

		Point3 support(const Point3& direction) const;

		Point3 center() const;

		void print(std::ostream& stream) const;

	protected:
		std::vector<Point3> _points;
	};



	class Simplex : public Polytope {
	public:
		Simplex(): Polytope(4) {
			_points.resize(4);
			_points[0] = Point3(0,0,0);
			_points[1] = Point3(1,0,0);
			_points[2] = Point3(0,1,0);
			_points[3] = Point3(0,0,1);
		}

		Simplex(const Point3& p1, const Point3& p2, const Point3& p3, const Point3& p4): Polytope(4) {
			_points.resize(4);
			_points[0] = p1;
			_points[1] = p2;
			_points[2] = p3;
			_points[3] = p4;
		}

		Simplex(const Simplex& other): Polytope(4){
			_points.resize(4);
			_points[0] = other[0];
			_points[1] = other[1];
			_points[2] = other[2];
			_points[3] = other[3];
		}

		Simplex(const Polytope& other): Polytope(4){
			_points.resize(4);
			_points[0] = other[0];
			_points[1] = other[1];
			_points[2] = other[2];
			_points[3] = other[3];
		}

		double volume() const;

		void orient();

		bool contains(const Point3& point) const;

		GeoVox::util::Point<4> barycentric(const Point3& point) const;
	};
}


#endif