#include "point.hpp"
#include <cmath>

namespace GeoVox::util
{
	//////////////////////////////////////////////////////////////////
	////////////////// DATA ACCESS METHODS ///////////////////////////
	//////////////////////////////////////////////////////////////////
	double& Point::operator[](int idx){
		return _data[idx];
	}

	double Point::operator[](int idx) const{
		return _data[idx];
	}


	//////////////////////////////////////////////////////////////////
	////////////////// ARITHMETIC METHODS ////////////////////////////
	//////////////////////////////////////////////////////////////////
	//additions
	Point* Point::operator+=(const Point& other){
		for (int i=0; i<_len; i++){
			_data[i] += other[i];
		}
		return this;
	}

	Point Point::operator+(const Point& other) const{
		Point result = Point(_len);
		for (int i=0; i<_len; i++){
			result[i] = _data[i] + other[i];
		}
		return result;
	}

	//subtractions
	Point* Point::operator-=(const Point& other){
		for (int i=0; i<_len; i++){
			_data[i] -= other[i];
		}
		return this;
	}

	Point Point::operator-(const Point& other) const{
		Point result = Point(_len);
		for (int i=0; i<_len; i++){
			result[i] = _data[i] - other[i];
		}
		return result;
	}

	//element-wise multiplication
	Point* Point::operator*=(const Point& other){
		for (int i=0; i<_len; i++){
			_data[i] *= other[i];
		}
		return this;
	}

	Point Point::operator*(const Point& other) const{
		Point result = Point(_len);
		for (int i=0; i<_len; i++){
			result[i] = _data[i] * other[i];
		}
		return result;
	}

	//element-wise division
	Point* Point::operator/=(const Point& other){
		for (int i=0; i<_len; i++){
			_data[i] /= other[i];
		}
		return this;
	}

	Point Point::operator/(const Point& other) const{
		Point result = Point(_len);
		for (int i=0; i<_len; i++){
			result[i] = _data[i] / other[i];
		}
		return result;
	}

	//scalar multiplication
	Point* Point::operator*=(const double& scalar){
		for (int i=0; i<_len; i++){
			_data[i] *= scalar;
		}
		return this;
	}

	Point Point::operator*(const double& scalar) const{
		Point result = Point(_len);
		for (int i=0; i<_len; i++){
			result[i] = _data[i]*scalar;
		}
		return result;
	}

	Point* Point::operator/=(const double& scalar){
		double reciprocal = 1.0/scalar;
		for (int i=0; i<_len; i++){
			_data[i] *= reciprocal;
		}
		return this;
	}

	Point Point::operator/(const double& scalar) const{
		double reciprocal = 1.0/scalar;
		Point result = Point(_len);
		for (int i=0; i<_len; i++){
			result[i] = _data[i]*reciprocal;
		}
		return result;
	}

	//////////////////////////////////////////////////////////////////
	////////////////// COMPARISION METHODS ///////////////////////////
	//////////////////////////////////////////////////////////////////
	bool Point::operator<(const Point& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] >= other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	bool Point::operator<=(const Point& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] > other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	bool Point::operator>(const Point& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] <= other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	bool Point::operator>=(const Point& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] < other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	bool Point::operator==(const Point& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] != other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	bool Point::operator!=(const Point& other) const{
		return !this->operator==(other);
	}

	//////////////////////////////////////////////////////////////////
	////////////////// CONVENIENCE METHODS ///////////////////////////
	//////////////////////////////////////////////////////////////////
	int Point::len() const{
		return this->_len;
	}

	void Point::fill(const double& scalar){//set all elements
		for (int i=0; i<_len; i++){
			_data[i] = scalar;
		}
	}

	double Point::max() const{ //get maximum element
		double maxval = _data[0];
		for (int i=0; i<_len; i++){
			if (_data[i] > maxval){
				maxval = _data[i];
			}
		}
		return maxval;
	}

	int Point::maxind() const{ //get lowest index of maximum element
		double maxval = _data[0];
		int maxind = 0;
		for (int i=0; i<_len; i++){
			if (_data[i] > maxval){
				maxval = _data[i];
				maxind = i;
			}
		}
		return maxind;
	}

	double Point::min() const{ //get minimum element
		double minval = _data[0];
		for (int i=0; i<_len; i++){
			if (_data[i] < minval){
				minval = _data[i];
			}
		}
		return minval;
	}

	int Point::minind() const{ //get lowest index of minimum element
		double minval = _data[0];
		int minind = 0;
		for (int i=0; i<_len; i++){
			if (_data[i] < minval){
				minval = _data[i];
				minind = i;
			}
		}
		return minind;
	}

	double Point::sum() const{
		double val = _data[0];
		for (int i=0; i<_len; i++){
			val += _data[i];
		}
		return val;
	}

	Point Point::normalize() const{
		double val = this->norm();
		Point result = operator/(val);
		return result;
	}


	//////////////////////////////////////////////////////////////////
	////////////////// DISTANCE METHODS /////////////////////////////
	//////////////////////////////////////////////////////////////////
	double Point::norm2() const{
		return this->dot(*this);
	}

	double Point::norm() const{
		return std::sqrt(this->norm2());
	}

	double Point::infnorm() const{
		double maxval = abs(_data[0]);
		double curval;
		for (int i=0; i<_len; i++){
			curval = abs(_data[i]);
			if (curval>maxval){
				maxval = curval;
			}
		}
		return maxval;
	}

	double Point::dist2(const Point& other) const{
		return (operator-(other)).norm2();
	}

	double Point::dist(const Point& other) const{
		return (operator-(other)).norm();
	}

	double Point::dist_inf(const Point& other) const{
		return (operator-(other)).infnorm();
	}

	//////////////////////////////////////////////////////////////////
	////////////////// LINEAR ALGEBRA METHODS ////////////////////////
	//////////////////////////////////////////////////////////////////
	double Point::dot(const Point& other) const{ //dot product
		double result = _data[0]*other[0];
		for (int i=0; i<_len; i++){
			result += _data[i]*other[i];
		}
		return result;
	}

	Point Point::cross(const Point& other) const{ //cross product, only if _len==3, but will return the cross product of the first 3 entries.
		Point result = Point(3);
		result[0] = _data[1]*other[2] - _data[2]*other[1];
		result[1] = _data[2]*other[0] - _data[0]*other[2];
		result[2] = _data[0]*other[1] - _data[1]*other[0];
		return result;
	}



	////////////// NON MEMBER FUNCTIONS IN namespace GeoVox:util

	Point el_max(const Point& point1, const Point& point2){ //element-wise maximum
		Point result = Point(point1.len());
		for (int i=0; i<point1.len(); i++){
			result[i] = std::max(point1[i], point2[i]);
		}
		return result;
	}
	Point el_min(const Point& point1, const Point& point2){ //element-wise minimum
		Point result = Point(point1.len());
		for (int i=0; i<point1.len(); i++){
			result[i] = std::min(point1[i], point2[i]);
		}
		return result;
	}

	Point  operator*(const double& scalar, const Point& rhs){return rhs*scalar;};

}