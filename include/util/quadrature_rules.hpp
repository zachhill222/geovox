#pragma once

#include<array>


namespace GV
{



//quadrature rules for the [-1,1] interval

template<int N, typename scalar_type=double> requires (N>0)
constexpr std::array<scalar_type,N> gauss_legendre_x()
{
	if constexpr (N==1) return {0.0};
	else if constexpr (N==2) return {-0.5773502691896257, 0.5773502691896257};
	else if constexpr (N==3) return {-0.7745966692414834, 0.0, 0.7745966692414834};
	else if constexpr (N==4) return {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
	else if constexpr (N==5) return {-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664};
	else {static_assert(N<=5, "Gauss-Legendre rules are only implemented for N<=5");}
	return std::array<scalar_type,N>{};
}

template<int N, typename scalar_type=double> requires (N>0)
constexpr std::array<scalar_type,N> gauss_legendre_w()
{
	if constexpr (N==1) return {2.0};
	else if constexpr (N==2) return {1.0, 1.0};
	else if constexpr (N==3) return {0.5555555555555556, 0.8888888888888889, 0.5555555555555556};
	else if constexpr (N==4) return {0.34785484513745385, 0.6521451548625462, 0.6521451548625462, 0.34785484513745385};
	else if constexpr (N==5) return {0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189};
	else {static_assert(N<=5, "Gauss-Legendre rules are only implemented for N<=5");}
	return std::array<scalar_type,N>{};
}


template<int BASE, int EXP>
constexpr int int_pow()
{
	int r=1;
	for (int i=0; i<EXP; ++i) {r*=BASE;}
	return r;
}

//helper functions to assemble cartesian product rules
//N- number of points along each axis
//D- spatial dimension
//C- requested coordinate
//quadrature points are (conceptually) arranged in a NxNxN... array (D-dimensional)
//the linear index into such an array is l = i + N*(j + N*k (...)) = i + N*j + N^2*k + ...
//this routine assembles the coordinate C of each of these points indexed by l above.
//note that l is a number in base-N with digits i, j, k, ... so that the index into the flat array
//is the Cth digit of l in this base. Note that (the integer part) of l/N^C is c + N*(...) (c=j above when N=1 and k when N=2)
//so c = (l/N^c) % N

template<int N, int D, int C, typename scalar_type=double>
	requires (N>0 && C < D && C >= 0)
constexpr std::array<scalar_type, int_pow<N,D>()> gauss_legendre_cartesian_coord_component()
{
	constexpr auto axis_x = gauss_legendre_x<N,scalar_type>();
	constexpr int cpow = int_pow<N,C>();

	std::array<scalar_type,int_pow<N,D>()> result;
	for (int l=0; l<int_pow<N,D>(); ++l) {
		result[l] = axis_x[(l/cpow) % N];
	}
	return result;
}

//this assembles the weights for the points in the NxN... array described above.
//note that we must take the product of the weights along each coordinate at each point
template<int N, int D, typename scalar_type=double>
	requires (N>0 && D>0)
constexpr std::array<scalar_type, int_pow<N,D>()> gauss_legendre_cartesian_weight()
{
	constexpr auto axis_w = gauss_legendre_w<N,scalar_type>();
	std::array<scalar_type,int_pow<N,D>()> result;
	result.fill(scalar_type{1.0});

	int cpow = 1;
	for (int c=0; c<D; ++c){
		for (int l=0; l<int_pow<N,D>(); ++l) {
			result[l] *= axis_w[(l/cpow) % N];
		}
		cpow *= N;
	}
	return result;
}

}