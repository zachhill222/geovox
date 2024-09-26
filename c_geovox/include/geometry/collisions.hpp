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


bool lineCase(Polytope& simplex, Point3& direction); //LINE CASE
bool triangleCase(Polytope& simplex, Point3& direction); //TRIANGLE CASE
bool tetraCase(Polytope& simplex, Point3& direction); //FULL SIMPLEX (TETRAHEDRON) CASE
bool doSimplex(Polytope& simplex, Point3& direction);



//IMPLEMENTATION
template<class SA, class SB> 
bool GJK(const SA& S1, const SB& S2){
	Point3 A, direction;
	
	// direction = Point3(-1,-1,-1);
	direction = S1.center() - S2.center();
	A = {support(S1,S2,direction)};
	
	Polytope simplex = Polytope({A});
	direction = -simplex[0];

	//MAIN LOOP
	for (int i=0; i<MAX_GJK_ITERATIONS; i++){
		A = support(S1,S2,direction);

		if (A.dot(direction) < 0){
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