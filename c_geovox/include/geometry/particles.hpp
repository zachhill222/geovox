#ifndef SUPERELLIPSOID_H
#define SUPERELLIPSOID_H

#include "util/point.hpp"
#include "util/quaternion.hpp"
#include "util/box.hpp"
#include "util/polytope.hpp"
#include "solvers/neldermead.hpp"

#include <functional>
#include <cmath>



using Point3 = GeoVox::util::Point<3>;
using Quaternion = GeoVox::util::Quaternion;
using Box = GeoVox::util::Box;
using Polytope3 = GeoVox::util::Polytope<3>;

typedef double (*NelderMeadFun_t)(GeoVox::util::Point<2>, GeoVox::util::Point<3>);

namespace GeoVox::geometry{

	class SuperEllipsoid{
	public:
		SuperEllipsoid(): _r(Point3(1,1,1)), _eps1(1), _eps2(1), _center(Point3(0,0,0)), _Q(Quaternion(1,0,0,0)) {
			_POWERS[0] = 1.0;
			_POWERS[1] = 1.0;
			_POWERS[2] = 1.0;

			_INVPOWERS[0] = 1.0;
			_INVPOWERS[1] = 1.0;
		}
		
		SuperEllipsoid(const Point3& r, const double& eps1, const double& eps2, const Point3& center, const Quaternion Q): _r(r), _eps1(eps1), _eps2(eps2), _center(center), _Q(Q) {
			_POWERS[0] = 1.0/eps1;
			_POWERS[1] = 1.0/eps2;
			_POWERS[2] = eps2/eps1;

			_INVPOWERS[0] = 1.0/(2.0-eps1);
			_INVPOWERS[1] = 1.0/(2.0-eps2);
		}

		//copy
		SuperEllipsoid copy() const;

		//get attributes
		Point3 center() const;
		Point3 radius() const;
		double eps1() const;
		double eps2() const;
		Quaternion quaternion() const;

		//bounds
		Box axis_alligned_bbox() const;
		Polytope3 bbox() const;

		//shifting
		SuperEllipsoid* operator+=(const Point3& other);
		SuperEllipsoid  operator+(const Point3& other) const;

		//rotating
		SuperEllipsoid* operator*=(const Quaternion& other);
		SuperEllipsoid  operator*(const Quaternion& other) const;

		//scaling
		SuperEllipsoid* operator*=(const double& other);
		SuperEllipsoid  operator*(const double& other) const;

		//comparison
		bool operator==(const SuperEllipsoid& other) const;
		inline bool operator!=(const SuperEllipsoid& other) const {return !(operator==(other));}

		//coordinate system changes
		Point3 tolocal(const Point3& point) const;
		Point3 toglobal(const Point3& point) const;
		
		//print
		void print(std::ostream &stream) const;

		//level set evaluation
		double levelval(const Point3& point) const;
		bool contains(const Point3& point) const;


		//convex support (point of tangency for a supporting hyperplane with normal unit vector direction)
		Point3 support(const Point3& direction) const;

		//get closest point on surface to given point
		Point3 closest_point(const Point3& point) const;

		//parametric coordinats: -pi/2 <= eta <= pi/2 and -pi <= omega <= pi SEE: http://www.cs.bilkent.edu.tr/~gudukbay/cs465/super_quadrics.pdf
		Point3 parametric(const double eta, const double omega) const; //get point from parametric coordinates
		Point3 normal_parametric(const double eta, const double omega) const; //get normal vector from parametric coordinates

		double neldermeadfun(GeoVox::util::Point<2> coords,  GeoVox::util::Point<3> localpoint) const;
		


	private:
		Point3 _r;
		double _eps1;
		double _eps2;
		Point3 _center;
		Quaternion _Q;
		double _POWERS[3];
		double _INVPOWERS[2];
	};





	class Sphere{
	public:
		Sphere(): _r(1), _center(Point3(0,0,0)) {
			_1r2 = 1.0/(_r*_r);
		}
		
		Sphere(const double r, const Point3& center): _r(r), _center(center) {
			_1r2 = 1.0/(_r*_r);
		}

		//copy
		Sphere copy() const;

		//get center
		Point3 center() const;

		//shifting
		Sphere* operator+=(const Point3& other);
		Sphere  operator+(const Point3& other) const;

		//rotating
		Sphere* operator*=(const Quaternion& other);
		Sphere  operator*(const Quaternion& other) const;

		//scaling
		Sphere* operator*=(const double& other);
		Sphere  operator*(const double& other) const;

		//coordinate system changes
		Point3 tolocal(const Point3& point) const;
		Point3 toglobal(const Point3& point) const;
		
		//print
		void print(std::ostream &stream) const;

		//level set evaluation
		double levelval(const Point3& point) const;

		//convex support (point of tangency for a supporting hyperplane with normal unit vector direction)
		Point3 support(const Point3& direction) const;

	private:
		double _r, _1r2;
		Point3 _center;
	};


	double cos_pow(const double theta, const double eps);
	double sin_pow(const double theta, const double eps);
}




#endif