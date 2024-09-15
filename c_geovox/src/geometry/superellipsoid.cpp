#include "geometry/superellipsoid.hpp"

namespace GeoVox::geometry{
	SuperEllipsoid SuperEllipsoid::copy() const{
		SuperEllipsoid result = SuperEllipsoid(_r, _eps1, _eps2, _center, _Q);
		return result;
	}

	SuperEllipsoid* SuperEllipsoid::operator+=(const GeoVox::util::Point& other){
		_center+= other;
		return this;
	}

	SuperEllipsoid SuperEllipsoid::operator+(const GeoVox::util::Point& other) const{
		SuperEllipsoid result = this->copy();
		result+=other;
		return result;
	}

	SuperEllipsoid* SuperEllipsoid::operator*=(const GeoVox::util::Quaternion& other){
		_Q*= other;
		return this;
	}

	SuperEllipsoid SuperEllipsoid::operator*(const GeoVox::util::Quaternion& other) const{
		SuperEllipsoid result = this->copy();
		result*=other;
		return result;
	}

	GeoVox::util::Point SuperEllipsoid::tolocal(const GeoVox::util::Point& point) const{
		GeoVox::util::Point result = _Q.rotate(point - _center);
		return result;
	}

	GeoVox::util::Point SuperEllipsoid::toglobal(const GeoVox::util::Point& point) const{
		GeoVox::util::Point result = _Q.conj().rotate(point) + _center;
		return result;
	}

	void SuperEllipsoid::print(std::ostream &stream) const{
		stream << "SuperEllipsoid( r=(" << _r[0] << ", " << _r[1] << ", " << _r[2] << "), eps1=" << _eps1 << ", eps2=" << _eps2 << ", center=(" << _center[0] << ", " << _center[1] << ", " << _center[2] << "), Q=(" << _Q[0] << ", " << _Q[1] << ", " << _Q[2] << ", " << _Q[3] << ") )\n";
	}

	double SuperEllipsoid::levelval(const GeoVox::util::Point& point) const{
		GeoVox::util::Point localpoint = this->tolocal(point);
		localpoint/=_r;
		localpoint*=localpoint;
		localpoint[0] = std::pow(localpoint[0], _POWERS[1]);
		localpoint[1] = std::pow(localpoint[1], _POWERS[1]);
		localpoint[2] = std::pow(localpoint[2], _POWERS[0]);
		return std::pow(localpoint[0]+localpoint[1], _POWERS[2]) + localpoint[2];
	}

}
