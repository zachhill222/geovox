#ifndef CONJUGATE_GRADIENT_H
#define CONJUGATE_GRADIENT_H

#define CONJUGATE_GRADIENT_EVAL_RESIDUAL 5
#define CONJUGATE_GRADIENT_RESIDUAL_TOL 1e-16

#include <vector>
#include <functional>

namespace GeoVox::solvers{
	template<typename Vec>
	double dot(const Vec &x, const Vec &y, long unsigned int start, long unsigned int end)
	{
		if (end-start < 128){
			double result = 0.0;
			for (long unsigned int i=start; i<end; i++){
				result += x[i]*y[i];
			}
			return result;
		}

		long unsigned int midway = (start+end)/2;
		//TODO: convert to parallel reduce
		return dot(x, y, start, midway) + dot(x, y, midway, end);
	}
	
	template<typename Vec>
	inline double dot(const Vec &x, Vec &y) {return dot(x, y, 0, x.size());} //return sum(x[i]*y[i])
	
	template<typename Vec>
	std::vector<double> sum_times(const Vec &x, const Vec &y, const double scalar) //return x[i] + scalar*y[i]
	{
		Vec result(x);
		for (long unsigned int i=0; i<x.size(); i++){
			result[i] += scalar*y[i];
		}
		return result;
	}


	template<typename Vec>
	void sum(Vec &x, const Vec &y) //inplace x[i] <- x[i] + y[i]
	{
		for (long unsigned int i=0; i<x.size(); i++){
			x[i] += y[i];
		}
	}

	template<typename Mat, typename Vec>
	void cg(Vec &x, const Mat &A, const Vec &b, const int n_iter)
	{
		//apply n_iter iterations of the conjugate gradient method
		//A must have the signature: std::vector<double> A(const std::vector<double>&)


		//INITIAL RESIDUAL
		std::vector<double> Ax = A(x);
		std::vector<double> r1 = sum_times(b, Ax, -1.0);
		if (dot(r1,r1)<CONJUGATE_GRADIENT_RESIDUAL_TOL){
			return;
		}

		//INITIAL SEARCH DIRECTION
		std::vector<double> p1(r1);

		//MAIN LOOP
		for (int i=0; i<n_iter; i++){
			std::vector<double> Ap1 = A(p1);
			double r1_norm2 = dot(r1,r1);
			double a = r1_norm2/dot(p1,Ap1);
			
			//update x
			x = sum_times(x, p1, a);
			
			//update residual
			std::vector<double> r2;
			double r2_norm2;
			if (i%CONJUGATE_GRADIENT_EVAL_RESIDUAL){
				Ax = A(x);
				r2 = sum_times(b, Ax, -1.0);
				r2_norm2 = dot(r2,r2);
				if (r2_norm2<CONJUGATE_GRADIENT_RESIDUAL_TOL){
					return;
				}
			}else{
				r2 = sum_times(r1, Ap1, -a);
				r2_norm2 = dot(r2,r2);
			}

			//update search direction
			double beta = r2_norm2/r1_norm2;
			p1 = sum_times(r2, p1, beta);
		}
	}



	

}


#endif