#ifndef BOX_H
#define BOX_H

#include "util/point.hpp"
#include <stdexcept>



namespace GeoVox::util{
	class Box{
	public:
		Box(): _low(Point(0.0, 0.0, 0.0)), _high(Point(1.0, 1.0, 1.0)) {}

		Box(const Point& vertex1, const Point& vertex2){
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

		~Box(){}

		//////////////////////////
		/////// ATTRIBUTES ///////
		//////////////////////////
		Point low() const;//get low
		Point high() const; //get high
		Point center() const; //get center of the box
		Point operator[](const int idx) const; //GET idx-TH VERTEX IN VTK-VOXEL ORDERING 
		void setlow(const Point& newlow);
		void sethigh(const Point& newhigh);
		Point hexvertex(const int idx) const; //Get idx-th vertex in vtk-hexahedral ordering


		///////////////////////////////////////////////
		/////// CONTAINMENT AND INTERSECTION //////////
		///////////////////////////////////////////////
		bool contains(const Point& point) const; //check if point is in box
		bool contains(const Box& other) const; //check if this box contains the entire other box
		bool intersects(const Box& other) const; //check if this box intersects the other box


		///////////////////////////////////////////////
		////////// SHIFTING AND SCALING ///////////////
		///////////////////////////////////////////////
		Box* operator+=(const Point& shift); //shift by +shift
		Box operator+(const Point& shift) const; 
		Box* operator-=(const Point& shift);
		Box operator-(const Point& shift) const;

		Box* operator*=(const double& scale); //scale towards center
		Box operator*(const double& scale) const;
		Box* operator/=(const double& scale);
		Box operator/(const double& scale) const;

		Box* combine(const Box& other); //combine this box with other (same thing as the Box(Box,Box) initializer)

	private:
		Point _low;
		Point _high;
	};
}




#endif




