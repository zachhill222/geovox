#include "geometry/polytope.hpp"


namespace GeoVox::geometry{
	Point Polytope::operator[](int idx) const{
		return _points[idx];
	}

	Point& Polytope::operator[](int idx){
		return _points[idx];
	}

	int Polytope::len() const{
		return _points.size();
	}

	void Polytope::addpoint(const Point& point){
		_points.push_back(point);
	}

	Point Polytope::support(const Point& direction) const{
		double maxdot = direction.dot(_points[0]);
		int maxind = 0;
		double tempdot;

		for (int i=1; i<len(); i++){
			tempdot = direction.dot(_points[i]);
			if (tempdot > maxdot){
				maxdot = tempdot;
				maxind = i;
			}
		}

		return _points[maxind];
	}

}