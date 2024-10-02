#include "mesh/vtk_linear_elements.hpp"

namespace GeoVox::mesh{
	std::vector<long unsigned int> Voxel::getface(const int face) const{
		switch (face){
		case 0:
			return std::vector<long unsigned int> {vertices[1], vertices[0], vertices[3], vertices[2]};
			break;
		case 1:
			return std::vector<long unsigned int> {vertices[0], vertices[1], vertices[4], vertices[5]};
			break;
		case 2:
			return std::vector<long unsigned int> {vertices[4], vertices[5], vertices[6], vertices[7]};
			break;
		case 3:
			return std::vector<long unsigned int> {vertices[6], vertices[7], vertices[2], vertices[3]};
			break;
		case 4:
			return std::vector<long unsigned int> {vertices[0], vertices[4], vertices[2], vertices[6]};
			break;
		case 5:
			return std::vector<long unsigned int> {vertices[5], vertices[1], vertices[7], vertices[3]};
			break;
		}
		return std::vector<long unsigned int> {};
	}

	std::vector<long unsigned int> Hexahedron::getface(const int face) const{
		switch (face){
		case 0:
			return std::vector<long unsigned int> {vertices[1], vertices[0], vertices[3], vertices[2]};
			break;
		case 1:
			return std::vector<long unsigned int> {vertices[0], vertices[1], vertices[5], vertices[4]};
			break;
		case 2:
			return std::vector<long unsigned int> {vertices[4], vertices[5], vertices[6], vertices[7]};
			break;
		case 3:
			return std::vector<long unsigned int> {vertices[7], vertices[6], vertices[2], vertices[3]};
			break;
		case 4:
			return std::vector<long unsigned int> {vertices[0], vertices[4], vertices[7], vertices[3]};
			break;
		case 5:
			return std::vector<long unsigned int> {vertices[5], vertices[1], vertices[2], vertices[6]};
			break;
		}
		return std::vector<long unsigned int> {};
	}

	std::vector<long unsigned int> Pixel::getface(const int face) const{
		switch (face){
		case 0:
			return std::vector<long unsigned int> {vertices[0], vertices[1]};
			break;
		case 1:
			return std::vector<long unsigned int> {vertices[1], vertices[3]};
			break;
		case 2:
			return std::vector<long unsigned int> {vertices[3], vertices[2]};
			break;
		case 3:
			return std::vector<long unsigned int> {vertices[2], vertices[0]};
			break;
		}
		return std::vector<long unsigned int> {};
	}

	std::vector<long unsigned int> Quad::getface(const int face) const{
		switch (face){
		case 0:
			return std::vector<long unsigned int> {vertices[0], vertices[1]};
			break;
		case 1:
			return std::vector<long unsigned int> {vertices[1], vertices[2]};
			break;
		case 2:
			return std::vector<long unsigned int> {vertices[2], vertices[3]};
			break;
		case 3:
			return std::vector<long unsigned int> {vertices[3], vertices[0]};
			break;
		}
		return std::vector<long unsigned int> {};
	}
}