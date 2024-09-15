#include "util/simplex.hpp"

namespace GeoVox::util{
	Point Simplex::operator[](int idx) const{
		return _vertex[idx];
	}

	Point& Simplex::operator[](int idx){
		return _vertex[idx];
	}
}