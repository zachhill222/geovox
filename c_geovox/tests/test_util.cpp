#include "util.hpp"
#include "geometry.hpp"
#include <iostream>

int test_point(){
	using namespace GeoVox;

	// initialize known 3-vectors
	util::Point u = util::Point(1,0,0);
	
	util::Point v = u;
	v[2]=2;

	u.print(std::cout);
	v.print(std::cout);
	u.cross(v).print(std::cout);

	geometry::SuperEllipsoid P = geometry::SuperEllipsoid();
	P.print(std::cout);
	P.tolocal(u);
	std::cout << P.levelval(u) << std::endl;

	util::Plane plane = util::Plane(u,v);
	plane.print(std::cout);

	plane.project(util::Point(4,5,6));

	util::Matrix3 mat = util::Matrix3();
	mat.setcol(0,u);
	mat.setcol(1,v);
	mat.setcol(2,u.cross(v));

	mat.print(std::cout);
	mat.inv().print(std::cout);
	(mat*mat.inv()).print(std::cout);


	return 1;
}



int main(int argc, char* argv[]){
	int flag = test_point();

	return flag;
}