#include "geometry/polytope.hpp"

namespace GeoVox::geometry{
	Point3 Polytope::operator[](int idx) const{
		return _points[idx];
	}

	Point3& Polytope::operator[](int idx){
		return _points[idx%_points.size()];
	}

	int Polytope::len() const{
		return _points.size();
	}

	void Polytope::addpoint(const Point3& point){
		_points.push_back(point);
	}

	Point3 Polytope::support(const Point3& direction) const{
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

	Point3 Polytope::center() const{
		double C = 1.0/_points.size();
		Point3 result = C*_points[0];

		for (long unsigned int i=0; i<_points.size(); i++){
			result += C*_points[i];
		}

		return result;
	}

	void Polytope::print(std::ostream& stream) const{
		for (int i=0; i<len(); i++){
			stream << i << ": ";
			_points[i].print(stream);
			stream << std::endl;
		}
	}





	//SIMPLEX
	double Simplex::volume() const{
		return ((_points[1]-_points[0]).cross(_points[2]-_points[0]).dot(_points[3]-_points[0]))/6.0 ;
	}

	void Simplex::orient(){
		if (volume() < 0){
			Point3 temp = _points[0];
			_points[0] = _points[1];
			_points[1] = temp;
		}
	}

	GeoVox::util::Point<4> Simplex::barycentric(const Point3& point) const{
		GeoVox::util::Point<4> coords = GeoVox::util::Point<4>();

		double vol = volume();
		for (int i=0; i<4; i++){
			Simplex newsimplex = *this;
			newsimplex[i] = point;
			coords[i] = newsimplex.volume()/vol;
		}

		return coords;
	}

	bool Simplex::contains(const Point3& point) const{
		GeoVox::util::Point<4> coords = barycentric(point);
		for (int i=0; i<4; i++){
			if (coords[i] < 0){
				return false;
			}
		}
		return true;
	}

}