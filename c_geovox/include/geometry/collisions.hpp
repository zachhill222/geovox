#ifndef COLLISIONS_H
#define COLLISIONS_H

#include "geometry/polytope.hpp"
#include "util/point.hpp"
#include "util/plane.hpp"
#include <cfloat> //need DBL_EPSILON

#define MAX_GJK_ITERATIONS 64

namespace GeoVox::geometry{
using Point3 = GeoVox::util::Point<3>;
using Simplex = GeoVox::geometry::Simplex;
using Polytope = GeoVox::geometry::Polytope;
using Plane = GeoVox::util::Plane;


//SA and SB are two classes that represent convex shapes with a methods:
//	Point support(const Point& direction)
//	Point center()

// FOR DETAILS, SEE:
// “Implementing GJK - 2006”
// by Casey Muratori
// and 
// https://cs.brown.edu/courses/cs195u/lectures/04_advancedCollisionsAndPhysics.pdf


//SUPPORT FUNCTION IN MINKOWSKI DIFFERENCE
template<class SA, class SB>
Point3 support(const SA& S1, const SB& S2, const Point3& direction){
	return S1.support(direction) - S2.support(-direction);
}


//LINE CASE
bool lineCase(Polytope& simplex, Point3& direction){
	Point3 &A = simplex[1]; //most recent point
	Point3 &B = simplex[0];

	Point3 AO = -A;
	Point3 AB = B-A;
	
	double DOT;

	DOT = AB.dot(AO);
	if (DOT>0.0){
		direction = AB.cross(AO.cross(AB));

		//check if line segment contained the origin. AB and AO are co-linear.
		if (direction.norm2() <= DBL_EPSILON){
			return true;
		}
		// simplex = Polytope({B, A}); //no change to simplex
	}
	else{
		direction = AO;
		simplex = Polytope({A});
	}


	return false;
}


//TRIANGLE CASE
bool triangleCase(Polytope& simplex, Point3& direction){
	Point3 &A = simplex[2]; //most recent point
	Point3 &B = simplex[1];
	Point3 &C = simplex[0];

	Point3 AO  = -A;
	Point3 AB  = B-A;
	Point3 AC  = C-A;

	Point3 ABC_normal = AB.cross(AC); //normal to triangle
	Point3 AB_normal  = AB.cross(ABC_normal); //away from triangle, normal to edge AB, in triangle plane
	Point3 AC_normal  = ABC_normal.cross(AC); //away from triangle, normal to edge AB, in triangle plane

	double DOT;

	DOT = AC_normal.dot(AO);
	if (DOT>0.0){
		DOT = AC.dot(AO);
		if (DOT>0.0){
			direction = AC.cross(AO.cross(AC));
			simplex = Polytope({C,A});

			// std::cout << "REGION 1\n";
		}
		else{
			DOT = AB.dot(AO);
			if (DOT>0.0){ //STAR
				direction = AB.cross(AO.cross(AB));
				simplex = Polytope({B, A});

				// std::cout << "REGION 3-\n";
			}
			else {
				direction = AO;
				simplex = Polytope({A});

				// std::cout << "REGION 2-\n";
			}
		}
	}
	else{
		DOT = AB_normal.dot(AO);
		if (DOT>0.0){
			DOT = AB.dot(AO);
			if (DOT>0.0){ //STAR
				direction = AB.cross(AO.cross(AB));
				simplex = Polytope({B,A});

				// std::cout << "REGION 3+\n";
			}
			else {
				direction = AO;
				simplex = Polytope({A});

				// std::cout << "REGION 2+\n";
			}
		}
		else{
			DOT = ABC_normal.dot(AO);
			// std::cout << DOT << std::endl;
			//above, below, or on triangle
			if (DOT>DBL_EPSILON ){
				direction = ABC_normal;
				// simplex = Polytope({C,B,A}); //no change to simplex

				// std::cout << "REGION 4\n";
			}
			else if (DOT<-DBL_EPSILON){
				direction = -ABC_normal;
				simplex = Polytope({B, C, A}); //orientation matters

				// std::cout << "REGION 5\n";
			}
			else{
				return true;
			}
		}
	}

	return false; //triangle PROBABLY doesn't contain the origin
}


//FULL SIMPLEX (TETRAHEDRON) CASE
bool tetraCase(Polytope& simplex, Point3& direction){
	Point3 &A = simplex[3]; //most recent point
	Point3 &B = simplex[2];
	Point3 &C = simplex[1];
	Point3 &D = simplex[0];

	Point3 O = Point3(0.0,0.0,0.0);

	Plane P;
	double abc, adc, abd;

	//get distance to each plane, we know the orighin is in the negative side of the plane BCD because A is the most recent point
	P   = Plane(A,B,C); //normal faces out of tetrahedron
	abc = P.dist(O);

	P   = Plane(A,D,C); //normal faces out of tetrahedron
	adc = P.dist(O);

	P   = Plane(A,B,D); //normal faces out of tetrahedron
	abd = P.dist(O);

	// if all distances are negative, origin is in side the tetrahedron
	double max_dist = std::max(abc,std::max(adc,abd));
	// std::cout << "max_dist= " << max_dist << std::endl;
	if (max_dist<0.0){
		return true;
	}


	// reduce to triangle case
	double abc_dist = abs(abc);
	double adc_dist = abs(adc);
	double abd_dist = abs(abd);
	double min_dist = std::min(abc_dist, std::min(adc_dist, abd_dist));
	// std::cout << "min_dist= " << min_dist << std::endl;

	if (abc_dist == min_dist){
		// std::cout << "Plane BCA\n";
		simplex = Polytope({B, C, A});
	}
	else if(adc_dist == min_dist){
		// std::cout << "Plane CDA\n";
		simplex = Polytope({C, D, A});
	}
	else{
		// std::cout << "Plane DBA\n";
		simplex = Polytope({D, B, A});
	}
	

	// run triangle case
	return triangleCase(simplex, direction);
}


bool doSimplex(Polytope& simplex, Point3& direction){
	//simplex must contain between 2 and 4 points initially
	//simplex and direction will both be updated for the next iteration

	bool result = false;
	


	//GET NEW SEARCH DIRECTION
	switch (simplex.len()){
	case 2:
		// std::cout << "LINE CASE\n";
		result = lineCase(simplex, direction);
		break;
	case 3:
		// std::cout << "TRIANGLE CASE\n";
		result = triangleCase(simplex, direction);
		break;
	case 4:
		// std::cout << "TETRAHEDRAL CASE\n";
		result = tetraCase(simplex, direction);
		break;
	}

	return result;
}



//IMPLEMENTATION
template<class SA, class SB> 
bool GJK(const SA& S1, const SB& S2){
	Point3 A, direction;
	
	direction = Point3(-1,-1,-1);
	// direction = S1.center() - S2.center();
	A = {support(S1,S2,direction)};
	
	Polytope simplex = Polytope({A});
	direction = -simplex[0];

	//MAIN LOOP
	for (int i=0; i<MAX_GJK_ITERATIONS; i++){
		A = support(S1,S2,direction);

		if (A.dot(direction) < 0.0){
			return false;
		}

		simplex.addpoint(A);

		// std::cout << "\n====================\niteration= " << i << "\tdirection= ";
		// direction.print(std::cout);
		// std::cout << "A.dot(direction)= " << A.dot(direction) << std::endl;
		// simplex.print(std::cout);


		if (doSimplex(simplex, direction)) {
			return true;
		}


		// std::cout << "\n====================\n";
	}

	return true; //failed to converge, return collision to be safe.
}


}

#endif