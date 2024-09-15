#ifndef POINT_H
#define POINT_H


#include <cmath>
#include <iostream>
#include <stdexcept>

namespace GeoVox::util{
	class Point{
	public:
		//////////////////////////////////////////////////////////////////
		////////////////// CONSTRUCTOR AND DESTRUCTOR ////////////////////
		//////////////////////////////////////////////////////////////////
		Point(): _len(3) {
			_data = new double[_len];
			_data[0] = 0.0;
			_data[1] = 0.0;
			_data[2] = 0.0;
		}

		Point(const int len): _len(len) {
			_data = new double[_len];
		}

		Point(const double x, const double y): _len(2){
			_data = new double[_len];
			_data[0] = x;
			_data[1] = y;
		}

		Point(const double x, const double y, const double z): _len(3){
			_data = new double[_len];
			_data[0] = x;
			_data[1] = y;
			_data[2] = z;
		}

		//copy constructor
		Point(const Point& other) {
			_len = other.len();
			_data = new double[_len];
			for (int i=0; i<_len; i++){
				_data[i] = other[i];
			}
		}

		~Point(){
			if (_data) {
				delete[] _data;
			}
		}

		
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
		Point cross(const Point& other) const; //cross product, only if _len==3, but no error check


	// private:
		double* _data;
		int _len;

	};


	//IMPLEMNT POINT-POINT OPERATORS IN GeoVox::utils NAMESPACE
	Point el_max(const Point& point1, const Point& point2); //element-wise maximum
	Point el_min(const Point& point1, const Point& point2); //element-wise minimum

	Point  operator*(const double& scalar, const Point& rhs);

}




#endif