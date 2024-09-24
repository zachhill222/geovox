#ifndef MATRIX_H
#define MATRIX_H

#include "util/point.hpp"
#include <iostream>

namespace GeoVox::util{
	using Point3 = Point<3>;
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
		Point3 solve(const Point3& rhs) const;

		Point3 col(int j) const;
		Point3 row(int i) const;
		void setcol(int j, const Point3& vec);
		void setrow(int i, const Point3& vec);

		Point3 operator*(const Point3& x) const;
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