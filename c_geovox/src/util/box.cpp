#include "util/box.hpp"

namespace GeoVox::util{

	Point3 Box::low() const{
		return _low;
	}

	Point3 Box::high() const{
		return _high;
	}

	Point3 Box::center() const{
		return 0.5*(_low+_high);
	}

	Point3 Box::operator[](const int idx) const{ //GET idx-TH VERTEX IN VTK-VOXEL ORDERING 
		switch(idx){
		case 0:
			return GeoVox::util::Point3(_low[0],  _low[1],  _low[2]);
			break;
		case 1:
			return GeoVox::util::Point3(_high[0], _low[1],  _low[2]);
			break;
		case 2:
			return GeoVox::util::Point3(_low[0],  _high[1], _low[2]);
			break;
		case 3:
			return GeoVox::util::Point3(_high[0], _high[1], _low[2]);
			break;
		case 4:
			return GeoVox::util::Point3(_low[0],  _low[1],  _high[2]);
			break;
		case 5:
			return GeoVox::util::Point3(_high[0], _low[1],  _high[2]);
			break;
		case 6:
			return GeoVox::util::Point3(_low[0],  _high[1], _high[2]);
			break;
		case 7:
			return GeoVox::util::Point3(_high[0], _high[1], _high[2]);
			break;	
		}
		std::out_of_range("Box: vertex index must be from 0 to 7");
		return Point3();
	}

	void Box::setlow(const Point3& newlow){
		Point3 _newlow = el_min(newlow, _high);
		Point3 _newhigh = el_max(newlow, _high);
		if (_newlow < _newhigh){
			_low = _newlow;
			_high = _newhigh;
		}else{
			std::out_of_range("Box: can't move _low to newlow");
		}
	}

	void Box::sethigh(const Point3& newhigh){
		Point3 _newlow = el_min(newhigh, _high);
		Point3 _newhigh = el_max(newhigh, _high);
		if (_newlow < _newhigh){
			_low = _newlow;
			_high = _newhigh;
		}else{
			std::out_of_range("Box: can't move _high to newhigh");
		}
	}

	Point3 Box::hexvertex(const int idx) const {
		switch (idx){
		case 2:
			return operator[](3);
			break;
		case 3:
			return operator[](2);
		case 6:
			return operator[](7);
			break;
		case 7:
			return operator[](6);
			break;
		default:
			return operator[](idx);
			break;
		}
		std::out_of_range("Box: vertex index must be from 0 to 7");
		return Point3();
	}

	bool Box::contains(const Point3& point) const{
		if (_low <= point && point <= _high){
			return true;
		}else{
			return false;
		}
	}

	bool Box::contains(const Box& other) const{
		if (_low <= other.low() && other.high() <= _high){
			return true;
		}else{
			return false;
		}
	}

	bool Box::intersects(const Box& other) const{ //return true if this box contains any vertex of the other box
		for (int i=0; i<8; i++){
			if (this->contains(other[i])){
				return true;
			}
		}
		for (int i=0; i<8; i++){
			if (other.contains(operator[](i))){
				return true;
			}
		}
		return false;
	}

	Point3 Box::support(const Point3& direction) const{
		double maxdot = direction.dot(operator[](0));
		int maxind = 0;
		double tempdot;

		for (int i=1; i<8; i++){
			tempdot = direction.dot(operator[](i));
			if (tempdot > maxdot){
				maxdot = tempdot;
				maxind = i;
			}
		}

		return operator[](maxind);
	}

	Box* Box::operator+=(const Point3& shift){
		_low+=shift;
		_high+=shift;
		return this;
	}

	Box Box::operator+(const Point3& shift) const{
		return Box(_low+shift, _high+shift);
	}

	Box* Box::operator-=(const Point3& shift){
		_low-=shift;
		_high-=shift;
		return this;
	}

	Box Box::operator-(const Point3& shift) const{
		return Box(_low-shift, _high-shift);
	}

	Box* Box::operator*=(const double& scale){
		Point3 _center = center();
		_low = _center + scale*(_low-_center);
		_high = _center + scale*(_high-_center);
		return this;
	}

	Box Box::operator*(const double& scale) const{
		Point3 _center = center();
		return Box(_center + scale*(_low-_center), _center + scale*(_high-_center));
	}

	Box* Box::operator/=(const double& scale){
		return this->operator*=(1.0/scale);
	}

	Box Box::operator/(const double& scale) const{
		return this->operator*(1.0/scale);
	}

	Box* Box::combine(const Box& other){
		Point3 _newlow = el_min(_low, other.low());
		Point3 _newhigh = el_max(_high, other.high());
		_low = _newlow;
		_high = _newhigh;
		return this;
	}


	Box operator*(const double& scale, const Box& box){
		Box result = box;
		result *= scale;
		return result;
	}



}