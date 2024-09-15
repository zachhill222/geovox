#ifndef COLLISIONS_H
#define COLLISIONS_H

#define MAX_GJK_ITERATIONS 64

#include "geometry/polytope.hpp"
#include "util/point.hpp"

namespace GeoVox::geometry{
using Point3 = GeoVox::util::Point<3>;
using Simplex = GeoVox::geometry::Simplex;

//A and B are two classes that represent convex shapes with a methods:
//	Point support(const Point& direction)
//	Point center()

template<class A, class B> 
bool GJK(const A& S1, const B& S2);



//IMPLEMENTATION
template<class A, class B> 
bool GJK(const A& S1, const B& S2){
	Point3 d; //search direction
	Point3 v0, v1, v2; //vertices and edges
	Point3 origin = Point3(0.0, 0.0, 0.0);

	Simplex simplex = Simplex(); //current simplex in the Minkowski difference


	//FILL OUT SIMPLEX
	//first point
	d = (S2.center() - S1.center()).normalize(); //starting search direction
	simplex[0] = S1.support(d) - S2.support(-d); //starting boundary point

	//second point
	d  = -simplex[0].normalize(); //new search direction
	v0 = S1.support(d) - S2.support(-d);
	if (v0.dot(d)<0){return false;}
	simplex[1] = v0;

	//third point
	v1 = simplex[0]-simplex[1];
	d  = v1.cross(-simplex[1]).cross(v1).normalize();
	v0 = S1.support(d) - S2.support(-d);
	if (v0.dot(d)<0){return false;}
	simplex[2] = v0;

	//fourth point
	v2 = simplex[2] - simplex[0];
	d  = v1.cross(v2).normalize();
	v0 = S1.support(d) - S2.support(-d);
	if (v0.dot(d)<0){return false;}
	simplex[3] = v0;

	//MAIN LOOP
	for (int i=0; i<MAX_GJK_ITERATIONS; i++){
		//get worst point
		double maxdist, tempdist;
		int maxind, tempind;

		maxdist = simplex[0].norm2();
		maxind = 0;

		for (int j=1; j<4; j++){
			tempdist = simplex[j].norm2();
			if (tempdist > maxdist){
				maxdist = tempdist;
				maxind  = j;
			}
		}


		//get new search direction
		v0 = simplex[maxind+1] - simplex[maxind+2];//simplex takes care of index looping
		v1 = simplex[maxind+1] - simplex[maxind+3];
		d  = v0.cross(v1).normalize();
		if (d.dot(simplex[maxind])>0){
			d = -d;
		}

		//get new vertex
		v0 = S1.support(d) - S2.support(-d);
		if (v0.dot(d)<0){return false;}
		simplex[maxind] = v0;

		//check if vertex contains origin
		if (simplex.contains(origin)){
			return true;
		}
	}

	return true; //failed to converge, return collision to be safe.
}
}

#endif