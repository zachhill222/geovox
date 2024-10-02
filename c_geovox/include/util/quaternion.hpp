#ifndef QUATERNION_H
#define QUATERNION_H

#include "util/box.hpp"
#include "util/point.hpp"
#include <cmath>

namespace GeoVox::util{
	using Point3 = Point<3>;
	class Quaternion{
	public:
		//// INITIALIZERS
		Quaternion(): _q0(1), _qv(Point3(0,0,0)) {}
		Quaternion(const double q0, const double q1, const double q2, const double q3): _q0(q0), _qv(Point3(q1,q2,q3)) {}
		Quaternion(const double q0, const Point3 qv): _q0(q0), _qv(Point3(qv[0], qv[1], qv[2])) {}

		//// ATTRIBUTES
		double operator[](int idx) const;
		double q0() const;
		Point3 qv() const;

		//// ROTATIONS
		Quaternion conj() const;
		Quaternion inv() const;
		double norm2() const;
		double norm() const;
		Quaternion* normalize(); //normalize this quaternion to a rotation quaternion
		Quaternion* setrotation(const double& theta, const Point3& axis);
		Point3 rotate(const Point3& point) const; //rotate point

		///// ARITHMETIC
		Quaternion* operator+=(const Quaternion& other);
		Quaternion operator+(const Quaternion& other) const;
		Quaternion* operator-=(const Quaternion& other);
		Quaternion operator-(const Quaternion& other) const;
		Quaternion* operator*=(const Quaternion& other);
		Quaternion operator*(const Quaternion& other) const;
		Quaternion* operator/=(const Quaternion& other);
		Quaternion operator/(const Quaternion& other) const;
		bool operator==(const Quaternion& other) const;
		inline bool operator!=(const Quaternion& other) const { return !(operator==(other));}

	private:
		double _q0;
		Point3 _qv;
	};
}


#endif