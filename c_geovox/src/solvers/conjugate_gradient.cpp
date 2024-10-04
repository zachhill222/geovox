// #include "solvers/conjugate_gradient.hpp"

// namespace GeoVox::solvers{
// 	std::vector<double> sum_times(const std::vector<double> &x, const std::vector<double> &y, const double scalar){
// 		std::vector<double> result(x);
// 		for (long unsigned int i=0; i<x.size(); i++){
// 			result[i] += scalar*y[i];
// 		}
// 		return result;
// 	}


// 	void sum(std::vector<double> &x, const std::vector<double> &y){
// 		for (long unsigned int i=0; i<x.size(); i++){
// 			x[i] += y[i];
// 		}
// 	}


// 	double dot(const std::vector<double> &x, const std::vector<double> &y, long unsigned int start, long unsigned int end){
// 		if (end-start < 128){
// 			double result = 0.0;
// 			for (long unsigned int i=start; i<end; i++){
// 				result += x[i]*y[i];
// 			}
// 			return result;
// 		}

// 		long unsigned int midway = (start+end)/2;
// 		//TODO: convert to parallel reduce
// 		return dot(x, y, start, midway) + dot(x, y, midway, end);
// 	}
// }