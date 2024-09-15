#include "util/matrix.hpp"

namespace GeoVox::util{
	double& Matrix3::operator[](int i){
		return _data[i];
	}

	double Matrix3::operator[](int i) const{
		return _data[i];
	}
	
	double& Matrix3::operator()(int i, int j){
		return _data[i+3*j]; //column-major for faster getting/setting column vectors
	}

	double Matrix3::operator()(int i, int j) const{
		return _data[3*i+j];
	}

	double Matrix3::det() const{
		return col(0).dot(col(1).cross(col(2)));
	}

	double Matrix3::trace() const{
		return _data[0] + _data[4] + _data[8];
	}

	Point Matrix3::col(int j) const{
		return Point(_data[3*j], _data[3*j+1], _data[3*j+2]);
	}

	Point Matrix3::row(int i) const{
		return Point(_data[i], _data[i+3], _data[i+6]);
	}

	void Matrix3::setcol(int j, const Point& vec){
		_data[3*j]   = vec[0];
		_data[3*j+1] = vec[1];
		_data[3*j+2] = vec[2];
	}

	void Matrix3::setrow(int i, const Point& vec){
		_data[i]   = vec[0];
		_data[i+3] = vec[1];
		_data[i+6] = vec[2];
	}

	Point Matrix3::solve(const Point& rhs) const{
		double C = 1.0/det();
		Point result = Point();
		Point col0 = col(0);
		Point col1 = col(1);
		Point col2 = col(2);

		result[0] = C*(col1.cross(col2).dot(rhs));
		result[1] = C*(col2.cross(col0).dot(rhs));
		result[2] = C*(col0.cross(col1).dot(rhs));

		return result;
	}


	Matrix3 Matrix3::inv() const{
		double C = 1.0/det();
		Matrix3 result = Matrix3();
		result.setrow(0, C*col(1).cross(col(2)));
		result.setrow(1, C*col(2).cross(col(0)));
		result.setrow(2, C*col(0).cross(col(1)));
		return result;
	}

	Point Matrix3::operator*(const Point& x) const{
		Point result = Point();
		result[0] = row(0).dot(x);
		result[1] = row(1).dot(x);
		result[2] = row(2).dot(x);
		return result;
	}

	Matrix3 Matrix3::operator*(const double c) const{
		Matrix3 result = *this;
		result*= c;
		return result;
	}

	Matrix3* Matrix3::operator*=(const double c){
		for (int i=0; i<9; i++){
			_data[i]*= c;
		}
		return this;
	}

	Matrix3 Matrix3::operator*(const Matrix3& other) const{
		Matrix3 result = Matrix3();
		result(0,0) = row(0).dot(other.col(0));
		result(1,0) = row(1).dot(other.col(0));
		result(2,0) = row(2).dot(other.col(0));
		result(0,1) = row(0).dot(other.col(1));
		result(1,1) = row(1).dot(other.col(1));
		result(2,1) = row(2).dot(other.col(1));
		result(0,2) = row(0).dot(other.col(2));
		result(1,2) = row(1).dot(other.col(2));
		result(2,2) = row(2).dot(other.col(2));
		return result;
	}

	Matrix3 Matrix3::operator+(const Matrix3& other) const{
		Matrix3 result = *this;
		result += other;
		return result;
	}

	Matrix3* Matrix3::operator+=(const Matrix3& other){
		for (int i=0; i<9; i++){
			_data[i] += other[i];
		}
		return this;
	}

	void Matrix3::print(std::ostream& stream) const{
		stream << _data[0] << " " << _data[3] << " " << _data[6] << std::endl;
		stream << _data[1] << " " << _data[4] << " " << _data[7] << std::endl;
		stream << _data[2] << " " << _data[5] << " " << _data[8] << std::endl;
	}
}