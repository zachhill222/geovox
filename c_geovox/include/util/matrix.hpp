#ifndef MATRIX_H
#define MATRIX_H

#include "util/point.hpp"
#include <iostream>

namespace GeoVox::util{
	class Matrix3{
	public:
		Matrix3(){}
		Matrix3(const Matrix3& other){
			for (int i=0; i<9; i++){
				_data[i] = other[i];
			}
		}

		double& operator[](int i);
		double operator[](int i) const;

		double& operator()(int i, int j);
		double operator()(int i, int j) const;

		double det() const;
		double trace() const;

		Matrix3 inv() const;
		Point solve(const Point& rhs) const;

		Point col(int j) const;
		Point row(int i) const;
		void setcol(int j, const Point& vec);
		void setrow(int i, const Point& vec);

		Point operator*(const Point& x) const;
		Matrix3 operator*(const double c) const;
		Matrix3 operator*(const Matrix3& other) const;
		Matrix3 operator+(const Matrix3& other) const;

		Matrix3* operator*=(const double c);
		Matrix3* operator+=(const Matrix3& other);

		void print(std::ostream& stream) const;

	private:
		double _data[9];
	};
}


#endif