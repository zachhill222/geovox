#pragma once
#include "box.hpp"
#include "point.hpp"
#include <cmath>

namespace GeoVox::util{
	class Quaternion{
		//// INITIALIZERS
		Quaternion(): _q0(1), _qv(Point(0,0,0)) {}
		Quaternion(const double q0, const double q1, const double q2, const double q3): _q0(q0), _qv(Point(q1,q2,q3)) {}
		Quaternion(const double q0, const Point qv): _q0(q0), _qv(Point(qv[0], qv[1], qv[2])) {}

	public:
		//// ATTRIBUTES
		double q0() const;
		Point qv() const;

		//// ROTATIONS
		Quaternion conj() const;
		Quaternion inv() const;
		double norm2() const;
		double norm() const;
		Quaternion* normalize(); //normalize this quaternion to a rotation quaternion
		Quaternion* setrotation(const double& theta, const Point& axis);
		Point rotate(const Point& point) const; //rotate point

		///// ARITHMETIC
		Quaternion* operator+=(const Quaternion& other);
		Quaternion operator+(const Quaternion& other) const;
		Quaternion* operator-=(const Quaternion& other);
		Quaternion operator-(const Quaternion& other) const;
		Quaternion* operator*=(const Quaternion& other);
		Quaternion operator*(const Quaternion& other) const;
		Quaternion* operator/=(const Quaternion& other);
		Quaternion operator/(const Quaternion& other) const;


	private:
		double _q0;
		Point _qv;
	};
}