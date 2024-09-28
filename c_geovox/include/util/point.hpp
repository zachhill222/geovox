#ifndef POINT_H
#define POINT_H


#include <cmath>
#include <iostream>
#include <stdexcept>

namespace GeoVox::util{
	template<size_t size>
	class Point{
	public:
		//////////////////////////////////////////////////////////////////
		////////////////// CONSTRUCTOR AND DESTRUCTOR ////////////////////
		//////////////////////////////////////////////////////////////////
		Point(){
			for (long unsigned int i=0; i<size; i++){
				_data[i] = 0.0;
			}
		}

		Point(const double x, const double y){
			_data[0] = x;
			_data[1] = y;
		}

		Point(const double x, const double y, const double z){
			_data[0] = x;
			_data[1] = y;
			_data[2] = z;
		}

		//copy constructor
		Point(const Point& other) {
			for (long unsigned int i=0; i<size; i++){
				_data[i] = other[i];
			}
		}

		// ~Point(){
		// 	if (_data) {
		// 		delete[] _data;
		// 	}
		// }

		
		//////////////////////////////////////////////////////////////////
		/////////////////// DATA ACCESS METHODS //////////////////////////
		//////////////////////////////////////////////////////////////////
		//access and set data
		double& operator[](int idx);
		double  operator[](int idx) const;


		//////////////////////////////////////////////////////////////////
		////////////////// ARITHMETIC METHODS ////////////////////////////
		//////////////////////////////////////////////////////////////////
		//negation
		Point operator-() const;

		//additions
		Point* operator+=(const Point& other);
		Point  operator+(const Point& other) const;

		//subtractions
		Point* operator-=(const Point& other);
		Point  operator-(const Point& other) const;

		//element-wise multiplication
		Point* operator*=(const Point& other);
		Point  operator*(const Point& other) const;

		//element-wise division
		Point* operator/=(const Point& other);
		Point  operator/(const Point& other) const;

		//scalar multiplication
		Point* operator*=(const double& scalar);
		Point  operator*(const double& scalar) const;
		Point* operator/=(const double& scalar);
		Point  operator/(const double& scalar) const;

		//copy
		Point& operator=(const Point& other);


		
		//////////////////////////////////////////////////////////////////
		//////////////// COMPARISION METHODS /////////////////////////////
		//////////////////////////////////////////////////////////////////
		bool operator<(const Point& other) const;
		bool operator<=(const Point& other) const;
		bool operator>(const Point& other) const;
		bool operator>=(const Point& other) const;
		bool operator==(const Point& other) const;
		bool operator!=(const Point& other) const;


		//////////////////////////////////////////////////////////////////
		/////////////// CONVENIENCE METHODS //////////////////////////////
		//////////////////////////////////////////////////////////////////
		int len() const; //get length
		void fill(const double& scalar); //set all elements to scalar
		double max() const; //get maximum element
		int maxind() const; //get lowest index of maximum element
		double min() const; //get minimum element
		int minind() const; //get lowest index of minimum element
		double sum() const; //sum all elements
		Point normalize() const; //normalize NOT in-place to a Euclidean norm of 1
		void print(std::ostream &stream) const; //print coordinate to ostream



		//////////////////////////////////////////////////////////////////
		////////////////// DISTANCE METHODS //////////////////////////////
		//////////////////////////////////////////////////////////////////
		double norm2() const;//Euclidean norm squared
		double norm() const;//Euclidean norm
		double infnorm() const;//max-norm
		double dist2(const Point& other) const; //Euclidean distance squared
		double dist(const Point& other) const; //Euclidean distance
		double dist_inf(const Point& other) const; //max-norm distance


		//////////////////////////////////////////////////////////////////
		////////////////// LINEAR ALGEBRA METHODS ////////////////////////
		//////////////////////////////////////////////////////////////////
		double dot(const Point& other) const; //dot product
		Point<3> cross(const Point<3>& other) const; //cross product, only if _len==3, but no error check


	private:
		double _data[size];
		static const int _len = size;

	};
	

	//IMPLEMENTATION
	template<size_t size>
	Point<size> el_max(const Point<size>& point1, const Point<size>& point2); //element-wise maximum

	template<size_t size>
	Point<size> el_min(const Point<size>& point1, const Point<size>& point2); //element-wise minimum

	template<size_t size>
	Point<size>  operator*(const double& scalar, const Point<size>& rhs);



	//////////////////////////////////////////////////////////////////
	////////////////// DATA ACCESS METHODS ///////////////////////////
	//////////////////////////////////////////////////////////////////
	template<size_t size>
	double& Point<size>::operator[](int idx){
		return _data[idx];
	}

	template<size_t size>
	double Point<size>::operator[](int idx) const{
		return _data[idx];
	}


	//////////////////////////////////////////////////////////////////
	////////////////// ARITHMETIC METHODS ////////////////////////////
	//////////////////////////////////////////////////////////////////
	template<size_t size>
	Point<size> Point<size>::operator-() const {
		Point<size> result = Point();
		for (int i=0; i<_len; i++){
			result[i] = -_data[i];
		}
		return result;
	}
	
	//additions
	template<size_t size>
	Point<size>* Point<size>::operator+=(const Point<size>& other){
		for (int i=0; i<_len; i++){
			_data[i] += other[i];
		}
		return this;
	}

	template<size_t size>
	Point<size> Point<size>::operator+(const Point<size>& other) const{
		Point<size> result = Point();
		for (int i=0; i<_len; i++){
			result[i] = _data[i] + other[i];
		}
		return result;
	}

	//subtractions
	template<size_t size>
	Point<size>* Point<size>::operator-=(const Point<size>& other){
		for (int i=0; i<_len; i++){
			_data[i] -= other[i];
		}
		return this;
	}

	template<size_t size>
	Point<size> Point<size>::operator-(const Point<size>& other) const{
		Point<size> result = Point();
		for (int i=0; i<_len; i++){
			result[i] = _data[i] - other[i];
		}
		return result;
	}

	//element-wise multiplication
	template<size_t size>
	Point<size>* Point<size>::operator*=(const Point<size>& other){
		for (int i=0; i<_len; i++){
			_data[i] *= other[i];
		}
		return this;
	}

	template<size_t size>
	Point<size> Point<size>::operator*(const Point<size>& other) const{
		Point<size> result = Point();
		for (int i=0; i<_len; i++){
			result[i] = _data[i] * other[i];
		}
		return result;
	}

	//element-wise division
	template<size_t size>
	Point<size>* Point<size>::operator/=(const Point<size>& other){
		for (int i=0; i<_len; i++){
			_data[i] /= other[i];
		}
		return this;
	}

	template<size_t size>
	Point<size> Point<size>::operator/(const Point<size>& other) const{
		Point<size> result = Point();
		for (int i=0; i<_len; i++){
			result[i] = _data[i] / other[i];
		}
		return result;
	}

	//scalar multiplication
	template<size_t size>
	Point<size>* Point<size>::operator*=(const double& scalar){
		for (int i=0; i<_len; i++){
			_data[i] *= scalar;
		}
		return this;
	}

	template<size_t size>
	Point<size> Point<size>::operator*(const double& scalar) const{
		Point<size> result = Point<size>();
		for (int i=0; i<_len; i++){
			result[i] = _data[i]*scalar;
		}
		return result;
	}

	template<size_t size>
	Point<size>* Point<size>::operator/=(const double& scalar){
		double reciprocal = 1.0/scalar;
		for (int i=0; i<_len; i++){
			_data[i] *= reciprocal;
		}
		return this;
	}

	template<size_t size>
	Point<size> Point<size>::operator/(const double& scalar) const{
		double reciprocal = 1.0/scalar;
		Point<size> result = Point();
		for (int i=0; i<_len; i++){
			result[i] = _data[i]*reciprocal;
		}
		return result;
	}

	template<size_t size>
	Point<size>& Point<size>::operator=(const Point<size>& other){
		if (this == &other){
			return *this;
		}

		if (_len != other.len()){
			std::out_of_range("Vectors must be the same size");
		}

		for (int i=0; i<_len; i++){
			_data[i] = other[i];
		}

		return *this;
	}

	//////////////////////////////////////////////////////////////////
	////////////////// COMPARISION METHODS ///////////////////////////
	//////////////////////////////////////////////////////////////////
	template<size_t size>
	bool Point<size>::operator<(const Point<size>& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] >= other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	template<size_t size>
	bool Point<size>::operator<=(const Point<size>& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] > other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	template<size_t size>
	bool Point<size>::operator>(const Point<size>& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] <= other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	template<size_t size>
	bool Point<size>::operator>=(const Point<size>& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] < other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	template<size_t size>
	bool Point<size>::operator==(const Point<size>& other) const{
		bool result = true;
		for (int i=0; i<_len; i++){
			if (_data[i] != other[i]){
				result = false;
				break;
			}
		}
		return result;
	}

	template<size_t size>
	bool Point<size>::operator!=(const Point<size>& other) const{
		return !this->operator==(other);
	}

	//////////////////////////////////////////////////////////////////
	////////////////// CONVENIENCE METHODS ///////////////////////////
	//////////////////////////////////////////////////////////////////
	template<size_t size>
	int Point<size>::len() const{
		return this->_len;
	}

	template<size_t size>
	void Point<size>::fill(const double& scalar){//set all elements
		for (int i=0; i<_len; i++){
			_data[i] = scalar;
		}
	}

	template<size_t size>
	double Point<size>::max() const{ //get maximum element
		double maxval = _data[0];
		for (int i=0; i<_len; i++){
			if (_data[i] > maxval){
				maxval = _data[i];
			}
		}
		return maxval;
	}

	template<size_t size>
	int Point<size>::maxind() const{ //get lowest index of maximum element
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

	template<size_t size>
	double Point<size>::min() const{ //get minimum element
		double minval = _data[0];
		for (int i=0; i<_len; i++){
			if (_data[i] < minval){
				minval = _data[i];
			}
		}
		return minval;
	}

	template<size_t size>
	int Point<size>::minind() const{ //get lowest index of minimum element
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

	template<size_t size>
	double Point<size>::sum() const{
		double val = _data[0];
		for (int i=0; i<_len; i++){
			val += _data[i];
		}
		return val;
	}

	template<size_t size>
	Point<size> Point<size>::normalize() const{
		double val = this->norm();
		Point<size> result = this->operator/(val);
		return result;
	}

	template<size_t size>
	void Point<size>::print(std::ostream &stream) const{
		for (int i=0; i<_len; i++){
			stream << _data[i] << " ";
		}
		// stream << "\n";
	}


	//////////////////////////////////////////////////////////////////
	////////////////// DISTANCE METHODS //////////////////////////////
	//////////////////////////////////////////////////////////////////
	template<size_t size>
	double Point<size>::norm2() const{
		return this->dot(*this);
	}

	template<size_t size>
	double Point<size>::norm() const{
		return std::sqrt(this->norm2());
	}

	template<size_t size>
	double Point<size>::infnorm() const{
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

	template<size_t size>
	double Point<size>::dist2(const Point<size>& other) const{
		return (operator-(other)).norm2();
	}

	template<size_t size>
	double Point<size>::dist(const Point<size>& other) const{
		return (operator-(other)).norm();
	}

	template<size_t size>
	double Point<size>::dist_inf(const Point<size>& other) const{
		return (operator-(other)).infnorm();
	}

	//////////////////////////////////////////////////////////////////
	////////////////// LINEAR ALGEBRA METHODS ////////////////////////
	//////////////////////////////////////////////////////////////////
	template<size_t size>
	double Point<size>::dot(const Point<size>& other) const{ //dot product
		double result = _data[0]*other[0];
		for (int i=1; i<_len; i++){
			result += _data[i]*other[i];
		}
		return result;
	}

	template<size_t size>
	Point<3> Point<size>::cross(const Point<3>& other) const{ //cross product, only if _len==3, but will return the cross product of the first 3 entries.
		Point<3> result = Point<3>();
		result[0] = _data[1]*other[2] - _data[2]*other[1];
		result[1] = _data[2]*other[0] - _data[0]*other[2];
		result[2] = _data[0]*other[1] - _data[1]*other[0];
		return result;
	}



	////////////// NON MEMBER FUNCTIONS IN namespace GeoVox:util
	template<size_t size>
	Point<size> el_max(const Point<size>& point1, const Point<size>& point2){ //element-wise maximum
		Point<size> result = Point<size>();
		for (int i=0; i<point1.len(); i++){
			result[i] = std::max(point1[i], point2[i]);
		}
		return result;
	}

	template<size_t size>
	Point<size> el_min(const Point<size>& point1, const Point<size>& point2){ //element-wise minimum
		Point<size> result = Point<size>();
		for (int i=0; i<point1.len(); i++){
			result[i] = std::min(point1[i], point2[i]);
		}
		return result;
	}

	template<size_t size>
	Point<size>  operator*(const double& scalar, const Point<size>& rhs){return rhs*scalar;};


}




#endif