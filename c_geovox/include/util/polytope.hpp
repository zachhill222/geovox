#ifndef POLYTOPES_H
#define POLYTOPES_H

#include "util/point.hpp"
#include <iostream>
#include <vector>
#include <initializer_list>


namespace GeoVox::util{
	//TYPE DEFS
	template <int dim>
	class Polytope;

	using Polytope3 = Polytope<3>;


	//POLYTOPE DEFINITION
	template<int dim>
	class Polytope{
	public:
		Polytope(const int npts){
			_points.reserve(npts);
		}

		Polytope(std::initializer_list<Point<dim>> list) : _points(list) {}

		Point<dim> operator[](int idx) const;
		Point<dim>& operator[](int idx);

		int len() const;

		void addpoint(const Point<dim>& point);

		Point<dim> support(const Point<dim>& direction) const;

		Point<dim> center() const;

		void print(std::ostream& stream) const;

	protected:
		std::vector<Point<dim>> _points;
	};


	//SIMPLEX DEFINITION
	template<int dim>
	class Simplex : public Polytope<dim> {
	public:
		Simplex(): Polytope<dim>(dim+1) {
			this->_points.push_back(Point<dim>()); //default constructor is all zeros
			for (long unsigned int i=1; i<dim+1; i++){
				this->_points.push_back(Point<dim>());
				this->_points[i][i-1] = 1.0;
			}
		}

		Simplex(std::initializer_list<Point<dim>> list): Polytope<dim>(list) {}

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
	template<int dim>
	Point<dim> Polytope<dim>::operator[](int idx) const{
		return _points[idx];
	}
	
	template<int dim>
	Point<dim>& Polytope<dim>::operator[](int idx){
		return _points[idx%_points.size()];
	}
	
	template<int dim>
	int Polytope<dim>::len() const{
		return _points.size();
	}
	
	template<int dim>
	void Polytope<dim>::addpoint(const Point<dim>& point){
		_points.push_back(point);
	}
	
	template<int dim>
	Point<dim> Polytope<dim>::support(const Point<dim>& direction) const{
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
	
	template<int dim>
	Point<dim> Polytope<dim>::center() const{
		double C = 1.0/_points.size();
		Point<dim> result = C*_points[0];

		for (long unsigned int i=0; i<_points.size(); i++){
			result += C*_points[i];
		}

		return result;
	}
	
	template<int dim>
	void Polytope<dim>::print(std::ostream& stream) const{
		for (int i=0; i<len(); i++){
			stream << i << ": ";
			_points[i].print(stream);
			stream << std::endl;
		}
	}
}


#endif