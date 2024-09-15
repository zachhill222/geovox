#include "util.hpp"
#include "geometry.hpp"
#include <iostream>


using namespace GeoVox;
using Point3 = util::Point<3>;
using Box = util::Box;

int test_point(){
	geometry::Simplex S = geometry::Simplex();
	S[0] = Point3(1,1,1);
	std::cout << "\n" << S.volume() << std::endl;
	S.print(std::cout);
	S.barycentric(Point3(.5,.5,2)).print(std::cout);


	std::cout << "orient S\n";
	S.orient();
	S.print(std::cout);
	S.barycentric(Point3(.5,.5,2)).print(std::cout);

	return 1;
}


int test_collision(){
	Box S1 = Box(Point3(0,0,0), Point3(1,1,1));
	Box S2 = 0.25*S1;
	bool collide = geometry::GJK(S1,S2);
	std::cout << "collide= " << collide << std::endl;
	std::cout << "intersect= " << S1.intersects(S2) << ", " << S2.intersects(S1) << std::endl;
	return 1;
}



int main(int argc, char* argv[]){
	int flag = test_collision();

	return flag;
}