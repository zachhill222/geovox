#ifndef BOX_H
#define BOX_H

#include "util/point.hpp"
#include <stdexcept>


using Point3 = GeoVox::util::Point<3>;

namespace GeoVox::util{
	class Box{
	public:
		Box(): _low(Point3(0.0, 0.0, 0.0)), _high(Point3(1.0, 1.0, 1.0)) {}

		Box(const Point3& vertex1, const Point3& vertex2){
			_low = el_min(vertex1, vertex2);
			_high = el_max(vertex1, vertex2);
			
			if (!(_low<_high)){
				std::invalid_argument("Box: low<high");
			}

		}

		Box(const Box& box1, const Box& box2){
			_low = el_min(box1.low(), box2.low());
			_high = el_max(box1.high(), box2.high());
			if (!(_low<_high)){
				std::invalid_argument("Box: low<high");
			}
		}

		Box(const Box& other){
			_low = other.low();
			_high = other.high();
		}


		//////////////////////////
		/////// ATTRIBUTES ///////
		//////////////////////////
		Point3 low() const;//get low
		Point3 high() const; //get high
		Point3 center() const; //get center of the box
		Point3 operator[](const int idx) const; //GET idx-TH VERTEX IN VTK-VOXEL ORDERING 
		void setlow(const Point3& newlow);
		void sethigh(const Point3& newhigh);
		Point3 hexvertex(const int idx) const; //Get idx-th vertex in vtk-hexahedral ordering


		///////////////////////////////////////////////
		/////// CONTAINMENT AND INTERSECTION //////////
		///////////////////////////////////////////////
		bool contains(const Point3& point) const; //check if point is in box
		bool contains_strict(const Point3& point) const; //check if point is in interior of the box
		bool contains(const Box& other) const; //check if this box contains the entire other box
		bool intersects(const Box& other) const; //check if this box intersects the other box
		Point3 support(const Point3& direction) const;


		///////////////////////////////////////////////
		////////// SHIFTING AND SCALING ///////////////
		///////////////////////////////////////////////
		Box* operator+=(const Point3& shift); //shift by +shift
		Box operator+(const Point3& shift) const; 
		Box* operator-=(const Point3& shift);
		Box operator-(const Point3& shift) const;

		Box* operator*=(const double& scale); //scale towards center
		Box operator*(const double& scale) const;
		Box* operator/=(const double& scale);
		Box operator/(const double& scale) const;

		Box* combine(const Box& other); //combine this box with other (same thing as the Box(Box,Box) initializer)

		void print(std::ostream &stream) const;

	private:
		Point3 _low;
		Point3 _high;
	};

	//LHS scalar multiplication
	Box operator*(const double& scale, const Box& box);
}




#endif




