#pragma once

#include "geometry/basic_shapes.hpp"
#include "geometry/point.hpp"
#include "geometry/plane.hpp"

#include <concepts>

#ifndef GUTIL_GJK_ZERO_TOL
	#define GUTIL_GJK_ZERO_TOL 1E-12
#endif

#ifndef GUTIL_MAX_GJK_ITERATIONS
	#define GUTIL_MAX_GJK_ITERATIONS 16
#endif

namespace gutil {
//SA and SB are two classes that represent convex shapes with a methods:
//	Point support(const Point& direction)
//	Point center()

// FOR DETAILS, SEE:
// “Implementing GJK - 2006”
// by Casey Muratori
// and 
// https://cs.brown.edu/courses/cs195u/lectures/04_advancedCollisionsAndPhysics.pdf

template<typename S, int DIM, typename T>
concept ShapeGJK = requires(const S& shape, const Point<DIM,T>& dir) {
	{shape.support(dir)} -> std::convertible_to<Point<DIM,T>>;
	{shape.center()} -> std::convertible_to<Point<DIM,T>>;
};


//SUPPORT FUNCTION IN MINKOWSKI DIFFERENCE
template<typename SA, typename SB, int DIM=3, typename T=double> requires ShapeGJK<SA,DIM,T> and ShapeGJK<SB,DIM,T>
Point<DIM,T> support(const SA& S1, const SB& S2, const Point<DIM,T>& direction)
{
	return static_cast<Point<DIM,T>>(S1.support(direction)) - static_cast<Point<DIM,T>>(S2.support(-direction));
}

//LINE CASE
template <int DIM=3, typename T=double>
bool lineCase(std::vector<Point<DIM,T>>& simplex, Point<DIM,T>& direction);

//TRIANGLE CASE
template <int DIM=3, typename T=double>
bool triangleCase(std::vector<Point<DIM,T>>& simplex, Point<DIM,T>& direction);

//FULL SIMPLEX (TETRAHEDRON) CASE
template <typename T=double>
bool tetraCase(std::vector<Point<3,T>>& simplex, Point<3,T>& direction);

//WRAPPER FUNCTION FOR SPECIAL CASES
template <int DIM=3, typename T=double>
bool doSimplex(std::vector<Point<DIM,T>>& simplex, Point<DIM,T>& direction);



//GJK IMPLEMENTATION
template<typename SA, typename SB, int DIM, typename T=double> requires ShapeGJK<SA,DIM,T> and ShapeGJK<SB,DIM,T>
bool collides_GJK(const SA& S1, const SB& S2)
{
	Point<DIM,T> direction = static_cast<Point<DIM,T>>(S1.center()) - static_cast<Point<DIM,T>>(S2.center());
	Point<DIM,T> A = support(S1,S2,direction);
	
	std::vector<Point<DIM,T>> simplex {A};
	direction = -simplex[0];

	//MAIN LOOP
	for (int i=0; i<GUTIL_MAX_GJK_ITERATIONS; i++){
		A = support(S1,S2,direction);

		if (dot(A,direction) < T{0}){
			return false;
		}

		simplex.push_back(A);

		if (doSimplex(simplex, direction)) {
			return true;
		}
	}

	return true; //failed to converge, return collision to be safe.
}


//LINE CASE IMPLEMENTATION
template <int DIM, typename T>
bool lineCase(std::vector<Point<DIM,T>>& simplex, Point<DIM,T>& direction)
{
	assert(simplex.size()==2);

	Point<DIM,T> &A = simplex[1]; //most recent point
	Point<DIM,T> &B = simplex[0];

	Point<DIM,T> AO = -A;
	Point<DIM,T> AB = B-A;
	
	double DOT;

	DOT = dot(AB,AO);
	if (DOT>T{0})
	{
		// std::cout << "AB\n";
		if constexpr (DIM==3){direction = cross(AB, cross(AO,AB));}
		else if constexpr (DIM==2) {
			Point<2,T> AB_perp{-AB[1], AB[0]};
			direction = dot(AB_perp, AO) < T{0} ? -AB_perp : AB_perp;
		}

		//check if line segment contained the origin. AB and AO are co-linear.
		if (squaredNorm(direction) <= GUTIL_GJK_ZERO_TOL)
		{
			return true;
		}
		// simplex = std::vector<Point<DIM,T>>({B, A}); //no change to simplex
	}
	else
	{
		direction = AO;
		simplex = std::vector<Point<DIM,T>>({A});
	}
	return false;
}


//TRIANGLE CASE IMPLEMENTATION
template <typename T>
bool triangleCase3(std::vector<Point<3,T>>& simplex, Point<3,T>& direction){
	assert(simplex.size()==3);

	Point<3,T> &A = simplex[2]; //most recent point
	Point<3,T> &B = simplex[1];
	Point<3,T> &C = simplex[0];

	Point<3,T> AO  = -A;
	Point<3,T> AB  = B-A;
	Point<3,T> AC  = C-A;

	Point<3,T> ABC_normal = cross(AB,AC); //normal to triangle
	Point<3,T> AB_normal  = cross(AB,ABC_normal); //away from triangle, normal to edge AB, in triangle plane
	Point<3,T> AC_normal  = cross(ABC_normal,AC); //away from triangle, normal to edge AB, in triangle plane

	double DOT;

	DOT = dot(AC_normal,AO);
	if (DOT>0.0){
		DOT = dot(AC,AO);
		if (DOT>0.0)
		{
			direction = cross(AC, cross(AO,AC));
			simplex = std::vector<Point<3,T>>({C,A});
		}
		else{
			DOT = dot(AB,AO);
			if (DOT>0.0)
			{ //STAR
				direction = cross(AB, cross(AO,AB));
				simplex = std::vector<Point<3,T>>({B, A});
			}
			else
			{
				direction = AO;
				simplex = std::vector<Point<3,T>>({A});
			}
		}
	}
	else{
		DOT = dot(AB_normal,AO);
		if (DOT>0.0){
			DOT = dot(AB,AO);
			if (DOT>0.0)
			{ //STAR
				direction = cross(AB, cross(AO,AB));
				simplex = std::vector<Point<3,T>>({B,A});
			}
			else
			{
				direction = AO;
				simplex = std::vector<Point<3,T>>({A});
			}
		}
		else
		{
			DOT = dot(ABC_normal,AO);
			//above, below, or on triangle
			if (DOT>GUTIL_GJK_ZERO_TOL)
			{
				direction = ABC_normal;
				// simplex = std::vector<Point<3,T>>({C,B,A}); //no change to simplex
			}
			else if (DOT<-GUTIL_GJK_ZERO_TOL)
			{
				direction = -ABC_normal;
				simplex = std::vector<Point<3,T>>({B, C, A}); //orientation matters
			}
			else {return true;}
		}
	}

	return false; //triangle PROBABLY doesn't contain the origin
}


//FULL SIMPLEX (TRIANGLE) CASE IMPLEMENTATION FOR 2D
template <typename T>
bool triangleCase2(std::vector<Point<2,T>>& simplex, Point<2,T>& direction)
{
	assert(simplex.size()==3);

	Point<2,T> &A = simplex[2]; //most recent point
	Point<2,T> &B = simplex[1];
	Point<2,T> &C = simplex[0];

	Point<2,T> AO = -A;
	Point<2,T> AB = B-A;
	Point<2,T> AC = C-A;

	// 2D perpendiculars (90 degree rotation)
	Point<2,T> AB_perp{-AB[1], AB[0]};  // perpendicular to AB
	Point<2,T> AC_perp{AC[1], -AC[0]};   // perpendicular to AC
	
	// Make sure perpendiculars point away from triangle
	// If AB_perp points toward C, flip it
	if (dot(AB_perp, AC) > T{0}) {
		AB_perp = -AB_perp;
	}
	if (dot(AC_perp, AB) > T{0}) {
		AC_perp = -AC_perp;
	}

	double DOT;

	// Check region near edge AC
	DOT = dot(AC_perp, AO);
	if (DOT > T{0}) {
		// Origin is on the AC side
		DOT = dot(AC, AO);
		if (DOT > T{0}) {
			// Origin is in AC region
			Point<2,T> AC_perp_to_O{-AC[1], AC[0]};
			if (dot(AC_perp_to_O, AO) < T{0}) {
				AC_perp_to_O = -AC_perp_to_O;
			}
			direction = AC_perp_to_O;
			simplex = std::vector<Point<2,T>>({C, A});
		} else {
			// Origin is in A region
			direction = AO;
			simplex = std::vector<Point<2,T>>({A});
		}
		return false;
	}

	// Check region near edge AB
	DOT = dot(AB_perp, AO);
	if (DOT > T{0}) {
		// Origin is on the AB side
		DOT = dot(AB, AO);
		if (DOT > T{0}) {
			// Origin is in AB region
			Point<2,T> AB_perp_to_O{-AB[1], AB[0]};
			if (dot(AB_perp_to_O, AO) < T{0}) {
				AB_perp_to_O = -AB_perp_to_O;
			}
			direction = AB_perp_to_O;
			simplex = std::vector<Point<2,T>>({B, A});
		} else {
			// Origin is in A region
			direction = AO;
			simplex = std::vector<Point<2,T>>({A});
		}
		return false;
	}

	// Origin is inside the triangle
	return true;
}


//FULL SIMPLEX (TETRAHEDRON) CASE IMPLEMENTATION FOR 3D
template <typename T>
bool tetraCase(std::vector<Point<3,T>>& simplex, Point<3,T>& direction)
{
	assert(simplex.size()==4);

	Point<3,T> &A = simplex[3]; //most recent point
	Point<3,T> &B = simplex[2];
	Point<3,T> &C = simplex[1];
	Point<3,T> &D = simplex[0];

	Point<3,T> O = Point<3,T> {0,0,0};

	Plane<T> P;
	T abc, adc, abd;

	//get distance to each plane, we know the orighin is in the negative side of the plane BCD because A is the most recent point
	P   = Plane<T>(A,B,C); //normal faces out of tetrahedron
	abc = P.dist(O);

	P   = Plane<T>(A,D,C); //normal faces out of tetrahedron
	adc = P.dist(O);

	P   = Plane<T>(A,B,D); //normal faces out of tetrahedron
	abd = P.dist(O);

	// if all distances are negative, origin is in side the tetrahedron
	T max_dist = max(abc,max(adc,abd));
	// std::cout << "max_dist= " << max_dist << std::endl;
	if (max_dist<0.0) {return true;}


	// reduce to triangle case
	T abc_dist = abs(abc);
	T adc_dist = abs(adc);
	T abd_dist = abs(abd);
	T min_dist = std::min(abc_dist, std::min(adc_dist, abd_dist));
	// std::cout << "min_dist= " << min_dist << std::endl;

	if (abc_dist == min_dist)
	{
		// std::cout << "Plane BCA\n";
		simplex = std::vector<Point<3,T>>({B, C, A});
	}
	else if(adc_dist == min_dist)
	{
		// std::cout << "Plane CDA\n";
		simplex = std::vector<Point<3,T>>({C, D, A});
	}
	else
	{
		// std::cout << "Plane DBA\n";
		simplex = std::vector<Point<3,T>>({D, B, A});
	}
	

	// run triangle case
	return triangleCase3(simplex, direction);
}


//WRAPPER IMPLEMENATION
template <int DIM, typename T>
bool doSimplex(std::vector<Point<DIM,T>>& simplex, Point<DIM,T>& direction)
{
	//simplex must contain between 2 and 4 points initially
	//simplex and direction will both be updated for the next iteration
	
	//GET NEW SEARCH DIRECTION
	switch (simplex.size()){
	case 2:
		// std::cout << "LINE CASE\n";
		return lineCase(simplex, direction);
	case 3:
		// std::cout << "TRIANGLE CASE\n";
		if constexpr (DIM==2) {return triangleCase2(simplex, direction);}
		else if constexpr (DIM==3) {return triangleCase3(simplex, direction);}
	case 4:
		// std::cout << "TETRAHEDRAL CASE\n";
		if constexpr (DIM==3) {return tetraCase(simplex, direction);}
	}

	throw std::runtime_error("doSimplex() - unkown simplex size");
	return false;
}
}
