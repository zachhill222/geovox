//these classes are for small dense matrices only
#pragma once

#ifndef GEOMETRY_UTILITY_ZERO_TOL
	//tolerance for determining if an element is essentially zero (e.g., in QR)
	#define GEOMETRY_UTILITY_ZERO_TOL 1E-10
#endif

#include "geometry/point.hpp"

#include <iostream>
#include <cassert>
#include <algorithm>

namespace gutil
{
	//Scalar_t should model the real line
	template <int n, int m, typename Scalar_t, bool COL_MAJOR=true>
	struct Matrix
	{
		Scalar_t _data[n*m];
		static constexpr int _index(const int i, const int j) noexcept {
			if constexpr (COL_MAJOR) {return n*j+i;}
			else {return m*i+j;}
		}

		using Row_t = Point<m,Scalar_t>;
		using Col_t = Point<n,Scalar_t>;
		static constexpr bool IS_SQUARE = n==m;

		constexpr Matrix() noexcept : _data{0} {}
		
		constexpr Matrix(const Scalar_t a) noexcept
		{
			for (int i=0; i<n; i++) {
				for (int j=0; j<m; j++) {
					if (i==j) {_data[_index(i,j)] = a;}
					else {_data[_index(i,j)] = Scalar_t{0};}
				}
			}
		}

		constexpr Matrix(const Matrix& other) noexcept
		{
			std::copy(other._data, other._data+n*m, _data);
		}

		constexpr Matrix(Matrix&& other) noexcept
		{
			std::move(other._data, other._data+n*m, _data);
		}

		//initializer list
		template<typename T>
		constexpr Matrix(std::initializer_list<std::initializer_list<T>> init) noexcept
		{
			int i=0;
			for (const auto& row : init) {
				assert(i<n);
				int j=0;
				for (const auto& val : row) {
					assert(j<m);
					(*this)(i, j) = static_cast<Scalar_t>(val);
					j++;
				}
				assert(j==m);
				i++;
			}
			assert(i==n);
		}

		//element access
		constexpr Scalar_t at(int i, int j) const
		{
			if (0 <= i and i < n) {
				if (0 <= j and j < m) {
					return _data[_index(i,j)];
				}
			}
			
			throw std::runtime_error("Matrix: index out of bounds");
		}

		constexpr Scalar_t& at(int i, int j)
		{
			if (0 <= i and i < n) {
				if (0 <= j and j < m) {
					return _data[_index(i,j)];
				}
			}
			
			throw std::runtime_error("Matrix: index out of bounds");
		}

		constexpr Scalar_t operator()(int i, int j) const noexcept
		{
			assert(0 <= i and i < n);
			assert(0 <= j and j < m);
			return _data[_index(i,j)];
		}

		constexpr Scalar_t& operator()(int i, int j) noexcept
		{
			assert(0 <= i and i < n);
			assert(0 <= j and j < m);
			return _data[_index(i,j)];
		}

		constexpr Scalar_t operator[](int k) const noexcept
		{
			assert(0 <= k and k<m*n);
			return _data[k];
		}

		constexpr Scalar_t& operator[](int k) noexcept
		{
			assert(0 <= k and k<m*n);
			return _data[k];
		}

		//copy and move assignment
		constexpr Matrix<n,m,Scalar_t,COL_MAJOR>& operator=(const Matrix<n,m,Scalar_t,COL_MAJOR>& other) noexcept
		{
			std::copy(other._data, other._data+n*m, _data);
			return *this;
		}

		constexpr Matrix<n,m,Scalar_t,COL_MAJOR>& operator=(Matrix<n,m,Scalar_t,COL_MAJOR>&& other) noexcept
		{
			std::move(other._data, other._data+n*m, _data);
			return *this;
		}

		//set identity along the main diagonal and zeros elsewhere
		constexpr void fill(const Scalar_t val) noexcept {std::fill(_data, _data+n*m, val);}

		//row and column access
		constexpr Row_t row(int i) const noexcept
		{
			assert(0<=i and i<n);
			Row_t result{};
			if constexpr (COL_MAJOR) {
				for (int j=0; j<m; j++) {
					result[j] = _data[n*j + i]; //_data[_index(i,j)]
				}
			}
			else
			{
				//row-major, all entries are contigous
				for (int j=0; j<m; j++) {
					result[j] = _data[m*i + j]; //_data[_index(i,j)]
				}
			}

			return result;
		}

		constexpr Col_t col(int j) const noexcept
		{
			assert(0<=j and j<m);
			Col_t result{};
			if constexpr (COL_MAJOR) {
				for (int i=0; i<n; i++) {
					result[i] = _data[n*j + i]; //_data[_index(i,j)]
				}
			}
			else
			{
				//row-major, all entries are contigous
				for (int i=0; i<n; i++) {
					result[i] = _data[m*i + j]; //_data[_index(i,j)]
				}
			}

			return result;
		}

		//transpose
		Matrix<m,n,Scalar_t,!COL_MAJOR> tr() const noexcept
		{
			Matrix<m,n,Scalar_t,!COL_MAJOR> result{};
			std::copy(_data, _data+n*m, result._data);
			return result;
		}

		//sum column vectors (sum across rows)
		constexpr Col_t row_sum() const noexcept
		{
			Col_t result(0);
			for (int i=0; i<n; i++) {
				for (int j=0; j<m; j++) {
					result[i] += (*this)(i,j);
				}
			}
		}

		//sum row vectors (sum up columns)
		constexpr Row_t col_sum() const noexcept
		{
			Row_t result(0);
			for (int i=0; i<n; i++) {
				for (int j=0; j<m; j++) {
					result[j] += (*this)(i,j);
				}
			}
		}

		//multiply column vector component-wise (multiply across rows)
		constexpr Col_t row_prod() const noexcept
		{
			Col_t result(1);
			for (int i=0; i<n; i++) {
				for (int j=0; j<m; j++) {
					result[i] *= (*this)(i,j);
				}
			}
		}

		//multiply row vectors component-wise (multiply up columns)
		constexpr Row_t col_prod() const noexcept
		{
			Row_t result(1);
			for (int i=0; i<n; i++) {
				for (int j=0; j<m; j++) {
					result[j] *= (*this)(i,j);
				}
			}
		}
	};


	///Print to ostream.
	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	std::ostream& operator<<(std::ostream& os, const Matrix<n,m,Scalar_t,COL_MAJOR> &matrix)
	{
		for (int i = 0; i < n; i++) {os << "[" << matrix.row(i) << "]\n";}
		return os;
	}

	//scalar-matrix multiplication
	template<int n, int m, typename Scalar_t, typename Scalar_u, bool COL_MAJOR>
	constexpr Matrix<n,m,Scalar_t,COL_MAJOR> operator*(const Scalar_u a, const Matrix<n,m,Scalar_t,COL_MAJOR> &matrix) noexcept
	{
		Matrix<n,m,Scalar_t,COL_MAJOR> result = matrix;
		for (int k=0; k<n*m; k++) {result[k] *= static_cast<Scalar_t>(a);}
		return result;
	}


	//matrix-vector multiplication
	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr Point<n,Scalar_t> operator*(const Matrix<n,m,Scalar_t,COL_MAJOR> &matrix, const Point<m,Scalar_t> &vector) noexcept
	{
		Point<n,Scalar_t> result{}; //all zeros
		for (int j=0; j<m; j++) {result += vector[j]*matrix.col(j);}
		return result;
	}

	//vector transpose * matrix
	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr Point<m,Scalar_t> operator*(const Point<n,Scalar_t> &vector, const Matrix<n,m,Scalar_t,COL_MAJOR> &matrix) noexcept
	{
		Point<m,Scalar_t> result{}; //all zeros
		for (int j=0; j<m; j++) {result[j] = dot(matrix.col(j),vector);}
		return result;
	}

	//vector-vector outer product
	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr Matrix<n,m,Scalar_t,COL_MAJOR> outer(const Point<n,Scalar_t> &left, const Point<m,Scalar_t> &right) noexcept
	{
		Matrix<n,m,Scalar_t,COL_MAJOR> result{};
		for (int i=0; i<n; i++)
		{
			for (int j=0; j<m; j++)
			{
				result(i,j) = left[i]*right[j];
			}
		}
		return result;
	}
	
	//matrix-matrix multiplication (same data order)
	template<int n, int m, int p, typename Scalar_t, bool COL_MAJOR>
	constexpr Matrix<n,p,Scalar_t,COL_MAJOR> operator*(const Matrix<n,m,Scalar_t,COL_MAJOR> &left, const Matrix<m,p,Scalar_t,COL_MAJOR> &right) noexcept
	{
		Matrix<n,p,Scalar_t,COL_MAJOR> result{};
		if constexpr (COL_MAJOR) {
			for (int j=0; j<p; j++) {
				for (int k=0; k<m; k++) {
					const Scalar_t r_kj = right._data[m*j+k]; //right(k,j)
					for (int i=0; i<n; i++) {
						//result(i,j)+=left(i,k) * right(k,j)
						result._data[n*j+i] += left._data[n*k+i] * r_kj;
					}
				}
			}
		} else {
			for (int i=0; i<n; i++) {
				for (int k=0; k<m; k++) {
					const Scalar_t l_ik = left._data[m*i+k]; //left(i,k)
					for (int j=0; j<p; j++) {
						//result(i,j)+=left(i,k) * right(k,j)
						result._data[p*i+j] += l_ik * right._data[p*k+j];
					}
				}
			}
		}

		return result;
	}

	//matrix-matrix multiplication (opposite data order)
	template<int n, int m, int p, typename Scalar_t, bool COL_MAJOR>
	constexpr Matrix<n,p,Scalar_t,COL_MAJOR> operator*(const Matrix<n,m,Scalar_t,COL_MAJOR> &left, const Matrix<m,p,Scalar_t,!COL_MAJOR> &right) noexcept
	{
		Matrix<n,p,Scalar_t,COL_MAJOR> result{};
		if constexpr (COL_MAJOR) {
			for (int j=0; j<p; j++) {
				for (int k=0; k<m; k++) {
					const Scalar_t r_kj = right._data[p*k+j]; //right(k,j)
					for (int i=0; i<n; i++) {
						//result(i,j)+=left(i,k) * right(k,j)
						result._data[n*j+i] += left._data[n*k+i] * r_kj;
					}
				}
			}
		} else {
			for (int i=0; i<n; i++) {
				for (int k=0; k<m; k++) {
					const Scalar_t l_ik = left._data[m*i+k]; //left(i,k)
					for (int j=0; j<p; j++) {
						//result(i,j)+=left(i,k) * right(k,j)
						result._data[p*i+j] += l_ik * right._data[m*k+j];
					}
				}
			}
		}

		return result;
	}

	//matrix least squares solution (must columns must be linearly independent)
	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr typename Matrix<n,m,Scalar_t,COL_MAJOR>::Row_t operator/(const Matrix<n,m,Scalar_t,COL_MAJOR> A, const typename Matrix<n,m,Scalar_t,COL_MAJOR>::Col_t &b) noexcept
	{
		Matrix<n,m,Scalar_t,COL_MAJOR> Q;
		Matrix<m,m,Scalar_t,COL_MAJOR> R;
		partialQR(A,Q,R);

		//note b*Q is b^T * Q = (Q.tr()*b)^T. For a vector, b and b^T are both the same C++ type
		return solve_upper(R,b*Q);
	}

	//matrix-matrix addition
	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr Matrix<n,m,Scalar_t,COL_MAJOR> operator+(const Matrix<n,m,Scalar_t,COL_MAJOR> &left, const Matrix<n,m,Scalar_t,COL_MAJOR> &right) noexcept
	{
		Matrix<n,m,Scalar_t,COL_MAJOR> result{};
		for (int k=0; k<n*m; k++) {result[k] = left[k]+right[k];}
		return result;
	}

	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr Matrix<n,m,Scalar_t,COL_MAJOR>& operator+=(Matrix<n,m,Scalar_t,COL_MAJOR> &left, const Matrix<n,m,Scalar_t,COL_MAJOR> &right) noexcept
	{
		for (int k=0; k<n*m; k++) {left[k]+=right[k];}
		return left;
	}

	//matrix-matrix subtraction
	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr Matrix<n,m,Scalar_t,COL_MAJOR> operator-(const Matrix<n,m,Scalar_t,COL_MAJOR> &left, const Matrix<n,m,Scalar_t,COL_MAJOR> &right) noexcept
	{
		Matrix<n,m,Scalar_t,COL_MAJOR> result{};
		for (int k=0; k<n*m; k++) {result[k] = left[k]-right[k];}
		return result;
	}

	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr Matrix<n,m,Scalar_t,COL_MAJOR>& operator-=(Matrix<n,m,Scalar_t,COL_MAJOR> &left, const Matrix<n,m,Scalar_t,COL_MAJOR> &right) noexcept
	{
		for (int k=0; k<n*m; k++) {left[k]-=right[k];}
		return left;
	}

	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr Matrix<n,m,Scalar_t,COL_MAJOR> operator-(const Matrix<n,m,Scalar_t,COL_MAJOR> &right) noexcept
	{
		Matrix<n,m,Scalar_t,COL_MAJOR> result{};
		for (int k=0; k<n*m; k++) {result[k]=-right[k];}
		return result;
	}

	//triangular matrix solve Ux=b with U upper triangular
	template<int n, typename Scalar_t, bool COL_MAJOR>
	constexpr Point<n,Scalar_t> solve_upper(const Matrix<n,n,Scalar_t,COL_MAJOR> &U, const Point<n,Scalar_t> &b) noexcept
	{
		Point<n,Scalar_t> x=b;
		for (int i=n-1; i>=0; i--)
		{
			assert(U(i,i)!=Scalar_t{0});

			for(int j=i+1; j<n; j++)
			{
				x[i] -= U(i,j)*x[j];
			}
			x[i]/= U(i,i);
		}
		return x;
	}

	//triangular matrix solve Lx=b with L lower triangular
	template<int n, typename Scalar_t, bool COL_MAJOR>
	constexpr Point<n,Scalar_t> solve_lower(const Matrix<n,n,Scalar_t,COL_MAJOR> &L, const Point<n,Scalar_t> &b) noexcept
	{
		Point<n,Scalar_t> x=b;
		for (int i=0; i<n; i++)
		{
			assert(L(i,i)!=Scalar_t{0});

			for(int j=0; j<i; j++)
			{
				x[i] -= L(i,j)*x[j];
			}
			x[i]/= L(i,i);
		}
		return x;
	}


	//Partial QR decomposition (modified Gram-Schmidt)
	template<int n, int m, typename Scalar_t, bool COL_MAJOR>
	constexpr void partialQR(const Matrix<n,m,Scalar_t,COL_MAJOR> &A, Matrix<n,m,Scalar_t,COL_MAJOR> &Q, Matrix<m,m,Scalar_t,COL_MAJOR> &R) noexcept
	{
		//Ensure that R and Q are zeros
		R = Matrix<m,m,Scalar_t,COL_MAJOR>(0);
		Q = A;

		//Modified G-S
		for (int k=0; k<m; k++)
		{
			for (int i=0; i<k; i++)
			{
				R(i,k) = dot(Q.col(k), Q.col(i));
				for (int j=0; j<n; j++) {Q(j,k)-= Q(j,i)*R(i,k);}
			}

			R(k,k) = norm2(Q.col(k));
			if (!(R(k,k)>GEOMETRY_UTILITY_ZERO_TOL)) {std::cout << A << std::endl;}
			assert(R(k,k)>GEOMETRY_UTILITY_ZERO_TOL);

			Scalar_t C = ((Scalar_t) 1)/R(k,k);
			for (int j=0; j<n; j++) {Q(j,k) *= C;}
		}
	}
}