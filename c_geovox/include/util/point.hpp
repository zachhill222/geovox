#ifndef POINT_H
#define POINT_H


#include "Eigen/Core"
#include "Eigen/Geometry"

#include <cmath>
namespace GeoVox::util{
	//////////////////////////////////////////////////////////////////
	////////////////// TYPEDEFS //////////////////////////////////////
	//////////////////////////////////////////////////////////////////
	template<int size>
	using Point  = Eigen::Matrix<double, 1, size>;

	using PointX = Eigen::RowVectorXd;
	using Point4 = Eigen::RowVector4d;
	using Point3 = Eigen::RowVector3d;
	using Point2 = Eigen::RowVector2d;

	//////////////////////////////////////////////////////////////////
	////////////////// COMPARISION FUNCTIONS (CONES) /////////////////
	//////////////////////////////////////////////////////////////////
	template <typename Derived>
	bool operator<(const Eigen::MatrixBase<Derived>& left, const Eigen::MatrixBase<Derived>& right){
		for (int i=0; i<left.size(); i++){
			if (left[i] >= right[i]){
				return false;
			}
		}
		return true;
	}

	template <typename Derived>
	bool operator<=(const Eigen::MatrixBase<Derived>& left, const Eigen::MatrixBase<Derived>& right){
		for (int i=0; i<left.size(); i++){
			if (left[i] > right[i]){
				return false;
			}
		}
		return true;
	}

	template <typename Derived>
	bool operator>(const Eigen::MatrixBase<Derived>& left, const Eigen::MatrixBase<Derived>& right) {
		return operator<(right, left);
	}

	template <typename Derived>
	bool operator>=(const Eigen::MatrixBase<Derived>& left, const Eigen::MatrixBase<Derived>& right) {
		return operator<=(right, left);
	}

	// template <typename Derived>
	// bool operator==(const Eigen::MatrixBase<Derived>& left, const Eigen::MatrixBase<Derived>& right){
	// 	for (int i=0; i<left.size(); i++){
	// 		if (left[i] != right[i]){
	// 			return false;
	// 		}
	// 	}
	// 	return true;
	// }


	//////////////////////////////////////////////////////////////////
	////////////////// ELEMENT-WISE FUNCTIONS ////////////////////////
	//////////////////////////////////////////////////////////////////
	// template <typename Derived>
	// typename Derived::PlainObject times(const Eigen::MatrixBase<Derived>& A, const Eigen::MatrixBase<Derived>& B){
	// 	return (A.array()*B.array()).matrix();
	// }

	// template <typename Derived>
	// typename Derived::PlainObject div(const Eigen::MatrixBase<Derived>& A, const Eigen::MatrixBase<Derived>& B){
	// 	return (A.array()/B.array()).matrix();
	// }


	//////////////////////////////////////////////////////////////////
	////////////////// UTILITY FUNCTIONS /////////////////////////////
	//////////////////////////////////////////////////////////////////

	template <typename Derived>
	typename Derived::PlainObject el_max(const Eigen::MatrixBase<Derived>& A, const Eigen::MatrixBase<Derived>& B){ //element-wise maximum
		Derived result;
		for (long int idx=0; idx<A.size(); idx++){
			result[idx] = (A[idx]>=B[idx]) ? (A[idx]) : (B[idx]);
		}
		return result;
	}

	template <typename Derived>
	typename Derived::PlainObject el_min(const Eigen::MatrixBase<Derived>& A, const Eigen::MatrixBase<Derived>& B){ //element-wise minimum
		Derived result;
		for (long int idx=0; idx<A.size(); idx++){
			result[idx] = (A[idx]<=B[idx]) ? (A[idx]) : (B[idx]);
		}
		return result;
	}


}




#endif