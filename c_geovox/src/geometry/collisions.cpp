#include "geometry/collisions.hpp"




namespace GeoVox::geometry{

//LINE CASE
bool lineCase(Polytope3& simplex, Point3& direction){
	Point3 &A = simplex[1]; //most recent point
	Point3 &B = simplex[0];

	Point3 AO = -A;
	Point3 AB = B-A;
	
	double DOT;

	DOT = AB.dot(AO);
	if (DOT>0.0){
		// std::cout << "AB\n";
		direction = AB.cross(AO.cross(AB));

		//check if line segment contained the origin. AB and AO are co-linear.
		if (direction.norm2() <= DBL_EPSILON){
			return true;
		}
		// simplex = Polytope3({B, A}); //no change to simplex
	}
	else{
		// std::cout << "AO\n";
		direction = AO;
		simplex = Polytope3({A});
	}


	return false;
}


//TRIANGLE CASE
bool triangleCase(Polytope3& simplex, Point3& direction){
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
			simplex = Polytope3({C,A});

			// std::cout << "REGION 1\n";
		}
		else{
			DOT = AB.dot(AO);
			if (DOT>0.0){ //STAR
				direction = AB.cross(AO.cross(AB));
				simplex = Polytope3({B, A});

				// std::cout << "REGION 3-\n";
			}
			else {
				direction = AO;
				simplex = Polytope3({A});

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
				simplex = Polytope3({B,A});

				// std::cout << "REGION 3+\n";
			}
			else {
				direction = AO;
				simplex = Polytope3({A});

				// std::cout << "REGION 2+\n";
			}
		}
		else{
			DOT = ABC_normal.dot(AO);
			// std::cout << DOT << std::endl;
			//above, below, or on triangle
			if (DOT>DBL_EPSILON ){
				direction = ABC_normal;
				// simplex = Polytope3({C,B,A}); //no change to simplex

				// std::cout << "REGION 4\n";
			}
			else if (DOT<-DBL_EPSILON){
				direction = -ABC_normal;
				simplex = Polytope3({B, C, A}); //orientation matters

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
bool tetraCase(Polytope3& simplex, Point3& direction){
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
		simplex = Polytope3({B, C, A});
	}
	else if(adc_dist == min_dist){
		// std::cout << "Plane CDA\n";
		simplex = Polytope3({C, D, A});
	}
	else{
		// std::cout << "Plane DBA\n";
		simplex = Polytope3({D, B, A});
	}
	

	// run triangle case
	return triangleCase(simplex, direction);
}


bool doSimplex(Polytope3& simplex, Point3& direction){
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


}