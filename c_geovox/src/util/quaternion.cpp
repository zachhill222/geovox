#include "util/quaternion.hpp"

namespace GeoVox::util{
	double Quaternion::operator[](int idx) const{
		if (idx==0){
			return _q0;
		}
		return _qv[idx-1];
	}

	double Quaternion::q0() const {return _q0;}

	Point Quaternion::qv() const {return _qv;}

	Quaternion Quaternion::conj() const{
		return Quaternion(_q0, -_qv);
	}
	Quaternion Quaternion::inv() const{
		double C = 1.0/norm2();
		return Quaternion(C*_q0, (-C)*_qv);
	}
	double Quaternion::norm2() const{
		return _q0*_q0 + _qv.norm2();
	}
	double Quaternion::norm() const{
		return std::sqrt(norm2());
	}

	Quaternion* Quaternion::normalize(){
		double C = 1.0/norm();
		_q0*=C;
		_qv*=C;
		return this;
	}
	Quaternion* Quaternion::setrotation(const double& theta, const Point& axis){
		_q0 = std::cos(0.5*theta);
		_qv = std::sin(0.5*theta)*axis.normalize();
		return this;
	}
	Point Quaternion::rotate(const Point& point) const {
		Quaternion V = Quaternion(0.0, point);
		V = operator*((V*conj()));
		return V.qv();
	}

	///// ARITHMETIC
	Quaternion* Quaternion::operator+=(const Quaternion& other){
		_q0+=other.q0();
		_qv+=other.qv();
		return this;
	}
	Quaternion Quaternion::operator+(const Quaternion& other) const{
		return Quaternion(_q0+other.q0(), _qv+other.qv());
	}
	Quaternion* Quaternion::operator-=(const Quaternion& other){
		_q0-=other.q0();
		_qv-=other.qv();
		return this;
	}
	Quaternion Quaternion::operator-(const Quaternion& other) const{
		return Quaternion(_q0-other.q0(), _qv-other.qv());
	}
	Quaternion* Quaternion::operator*=(const Quaternion& other){
		double Q0 = _q0*other.q0() - _qv.dot(other.qv());
		_qv = _q0*other.qv() + other.q0()*_qv + _qv.cross(other.qv());
		_q0 = Q0;
		return this;
	}
	Quaternion Quaternion::operator*(const Quaternion& other) const{
		double Q0 = _q0*other.q0() - _qv.dot(other.qv());
		Point  QV = _q0*other.qv() + other.q0()*_qv + _qv.cross(other.qv());
		return Quaternion(Q0, QV);
	}
	Quaternion* Quaternion::operator/=(const Quaternion& other){
		operator*=(other.inv());
		return this;
	}
	Quaternion Quaternion::operator/(const Quaternion& other) const{
		return operator*(other.inv());
	}


}