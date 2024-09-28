#ifndef POLYTOPES_H
#define POLYTOPES_H

#include "util/point.hpp"
#include <iostream>
#include <vector>
#include <initializer_list>


namespace GeoVox::util{
	
	template<size_t dim>
	class Polytope{
	public:
		Polytope(const int npts){
			_points.reserve(npts);
		}

		Polytope(std::initializer_list<GeoVox::util::Point<dim>> list) : _points(list) {}

		GeoVox::util::Point<dim> operator[](int idx) const;
		GeoVox::util::Point<dim>& operator[](int idx);

		int len() const;

		void addpoint(const GeoVox::util::Point<dim>& point);

		GeoVox::util::Point<dim> support(const GeoVox::util::Point<dim>& direction) const;

		GeoVox::util::Point<dim> center() const;

		void print(std::ostream& stream) const;

	protected:
		std::vector<GeoVox::util::Point<dim>> _points;
	};


	template<size_t dim>
	class Simplex : public Polytope<dim> {
	public:
		Simplex(): Polytope<dim>(dim+1) {
			this->_points.push_back(GeoVox::util::Point<dim>()); //default constructor is all zeros
			for (long unsigned int i=1; i<dim+1; i++){
				this->_points.push_back(GeoVox::util::Point<dim>());
				this->_points[i][i-1] = 1.0;
			}
		}

		Simplex(std::initializer_list<GeoVox::util::Point<dim>> list): Polytope<dim>(list) {}

		Simplex(const Simplex<dim>& other): Polytope<dim>(dim+1){
			for (int i=0; i<other.len(); i++){
				this->_points.push_back(other[i]);
			}
		}

		Simplex(const Polytope<dim>& other): Polytope<dim>(dim+1){
			for (int i=0; i<other.len(); i++){
				this->_points.push_back(other[i]);
			}
		}
	};





	//POLYTOPE IMPLEMENTATION
	template<size_t dim>
	GeoVox::util::Point<dim> Polytope<dim>::operator[](int idx) const{
		return _points[idx];
	}
	
	template<size_t dim>
	GeoVox::util::Point<dim>& Polytope<dim>::operator[](int idx){
		return _points[idx%_points.size()];
	}
	
	template<size_t dim>
	int Polytope<dim>::len() const{
		return _points.size();
	}
	
	template<size_t dim>
	void Polytope<dim>::addpoint(const GeoVox::util::Point<dim>& point){
		_points.push_back(point);
	}
	
	template<size_t dim>
	GeoVox::util::Point<dim> Polytope<dim>::support(const GeoVox::util::Point<dim>& direction) const{
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
	
	template<size_t dim>
	GeoVox::util::Point<dim> Polytope<dim>::center() const{
		double C = 1.0/_points.size();
		GeoVox::util::Point<dim> result = C*_points[0];

		for (long unsigned int i=0; i<_points.size(); i++){
			result += C*_points[i];
		}

		return result;
	}
	
	template<size_t dim>
	void Polytope<dim>::print(std::ostream& stream) const{
		for (int i=0; i<len(); i++){
			stream << i << ": ";
			_points[i].print(stream);
			stream << std::endl;
		}
	}
}


#endif