#include "util.hpp"
#include "geometry.hpp"
#include <iostream>


using namespace GeoVox;
using Point3 = util::Point<3>;
using Box = util::Box;
using Sphere = geometry::Sphere;
using SuperEllipsoid = geometry::SuperEllipsoid;
using Assembly = geometry::Assembly;


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
	Box S1 = Box(Point3(-1,-1,-1), Point3(1,1,1));
	Box S2 = S1+2*Point3(1,1,0);

	Sphere S3 = Sphere(1.7321, Point3(0,0,0));
	Sphere S4 = S3*0.5 + Point3(1,0,0);

	SuperEllipsoid S5 = SuperEllipsoid();


	bool collide = geometry::GJK(S1,S5+Point3(1,0,0));
	std::cout << "collide= " << collide << std::endl;
	return 1;
}


int test_assembly(){
	Assembly A = Assembly("particles_50.txt");
	A.maxdepth(5);

	// A.print(std::cout);
	A.make_tree();
	A.print_tree(std::cout);
	return 1;
}

int main(int argc, char* argv[]){
	// int flag = test_collision();
	int flag = test_assembly();
	return flag;
}