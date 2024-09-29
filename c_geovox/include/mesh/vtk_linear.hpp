#ifndef VTK_LINEAR_H
#define VTK_LINEAR_H

#include "util/point.hpp"
#include "util/box.hpp"

#include <set>
#include <vector>
#include <initializer_list>

using Point3 = GeoVox::util::Point<3>;


namespace GeoVox::mesh{
	class Vertex{
	public:
		Vertex(){
			coords[0] = 0.0;
			coords[1] = 0.0;
			coords[2] = 0.0;
		}

		Vertex(const double x, const double y, const double z) {
			coords[0] = x;
			coords[1] = y;
			coords[2] = z;
		}

		Vertex(const Point3& point){
			coords[0] = point[0];
			coords[1] = point[1];
			coords[2] = point[2];
		}

		Vertex(const Vertex& other){
			coords[0] = other[0];
			coords[1] = other[1];
			coords[2] = other[2];
		}

		double coords[3];
		static const int vtkID = 1;

		inline double& operator[](int idx){return coords[idx];}
		inline double operator[](int idx) const{return coords[idx];}
		inline Point3 topoint() const{return Point3(coords[0], coords[1], coords[2]);}

		bool operator==(const Vertex& other);
		bool operator<=(const Vertex& other);
		bool operator<(const Vertex& other);
	};


	template<int size>
	class Element{
	public:
		Element(){
			vertices.resize(size);
		}

		Element(const std::initializer_list<long unsigned int>& list) : vertices(list){}

		Element(const std::vector<long unsigned int>& list) {
			vertices.reserve(size);
			for (long unsigned int i=0; i<size; i++){
				vertices.push_back(list[i]);
			}
		}

		std::vector<long unsigned int> vertices;
		inline long unsigned int& operator[](int idx){return vertices[idx];}
		inline long unsigned int operator[](int idx) const {return vertices[idx];}
	};


	class Voxel : public Element<8>{
	public:
		Voxel(const std::initializer_list<long unsigned int>& list) : Element(list) {}
		Voxel(const std::vector<long unsigned int>& list) : Element(list) {}

		static const int vtkID = 11;
		std::vector<long unsigned int> getface(const int face) const;
	};


	class Hexahedron : public Element<8>{
	public:
		Hexahedron(const std::initializer_list<long unsigned int>& list) : Element(list) {}
		Hexahedron(const std::vector<long unsigned int>& list) : Element(list) {}

		Hexahedron(const Voxel& voxel){
			vertices[0] = voxel[0];
			vertices[1] = voxel[1];
			vertices[2] = voxel[3];
			vertices[3] = voxel[2];
			vertices[4] = voxel[4];
			vertices[5] = voxel[5];
			vertices[6] = voxel[7];
			vertices[7] = voxel[6];
		}

		static const int vtkID = 12;
		std::vector<long unsigned int> getface(const int face) const;
	};



	class Pixel : public Element<4>{
		Pixel(const std::initializer_list<long unsigned int>& list) : Element(list) {}
		Pixel(const std::vector<long unsigned int>& list) : Element(list) {}

		Pixel(const Voxel& voxel, const int face){
			switch (face){
			case 0:
				vertices[0] = voxel[1];
				vertices[1] = voxel[0];
				vertices[2] = voxel[3];
				vertices[3] = voxel[2];
				break;
			case 1:
				vertices[0] = voxel[0];
				vertices[1] = voxel[1];
				vertices[2] = voxel[4];
				vertices[3] = voxel[5];
				break;
			case 2:
				vertices[0] = voxel[4];
				vertices[1] = voxel[5];
				vertices[2] = voxel[6];
				vertices[3] = voxel[7];
				break;
			case 3:
				vertices[0] = voxel[6];
				vertices[1] = voxel[7];
				vertices[2] = voxel[2];
				vertices[3] = voxel[3];
				break;
			case 4:
				vertices[0] = voxel[0];
				vertices[1] = voxel[4];
				vertices[2] = voxel[2];
				vertices[3] = voxel[6];
				break;
			case 5:
				vertices[0] = voxel[5];
				vertices[1] = voxel[1];
				vertices[2] = voxel[7];
				vertices[3] = voxel[3];
				break;
			}
		}

		Pixel(const Voxel& voxel, const std::set<int>& vertex_set){
			if (vertex_set == std::set{0,1,2,3}){
				Pixel(voxel, 0);
			} else if (vertex_set == std::set{0,1,4,5}){
				Pixel(voxel, 1);
			} else if (vertex_set == std::set{4,5,6,7}){
				Pixel(voxel, 2);
			} else if (vertex_set == std::set{2,3,6,7}){
				Pixel(voxel, 3);
			} else if (vertex_set == std::set{0,2,4,6}){
				Pixel(voxel, 4);
			} else if (vertex_set == std::set{1,3,5,7}){
				Pixel(voxel, 5);
			}
		}

		static const int vtkID = 8;
		std::vector<long unsigned int> getface(const int face) const;
	};


	class Quad : public Element<4>{
		Quad(const std::initializer_list<long unsigned int>& list) : Element(list) {}
		Quad(const std::vector<long unsigned int>& list) : Element(list) {}

		Quad(const Hexahedron& hex, const int face){
			switch (face){
			case 0:
				vertices[0] = hex[1];
				vertices[1] = hex[0];
				vertices[2] = hex[3];
				vertices[3] = hex[2];
				break;
			case 1:
				vertices[0] = hex[0];
				vertices[1] = hex[1];
				vertices[2] = hex[5];
				vertices[3] = hex[4];
				break;
			case 2:
				vertices[0] = hex[4];
				vertices[1] = hex[5];
				vertices[2] = hex[6];
				vertices[3] = hex[7];
				break;
			case 3:
				vertices[0] = hex[7];
				vertices[1] = hex[6];
				vertices[2] = hex[2];
				vertices[3] = hex[3];
				break;
			case 4:
				vertices[0] = hex[0];
				vertices[1] = hex[4];
				vertices[2] = hex[7];
				vertices[3] = hex[3];
				break;
			case 5:
				vertices[0] = hex[5];
				vertices[1] = hex[1];
				vertices[2] = hex[2];
				vertices[3] = hex[6];
				break;
			}
		}

		Quad(const Hexahedron& hex, const std::set<int>& vertex_set){
			if (vertex_set == std::set{0,1,2,3}){
				Quad(hex, 0);
			} else if (vertex_set == std::set{0,1,4,5}){
				Quad(hex, 1);
			} else if (vertex_set == std::set{4,5,6,7}){
				Quad(hex, 2);
			} else if (vertex_set == std::set{2,3,6,7}){
				Quad(hex, 3);
			} else if (vertex_set == std::set{0,3,4,7}){
				Quad(hex, 4);
			} else if (vertex_set == std::set{1,2,5,6}){
				Quad(hex, 5);
			}
		}

		Quad(const Voxel& voxel, const int face){
			Quad(Hexahedron(voxel), face);
		}

		Quad(const Voxel& voxel, const std::set<int>& vertex_set){
			Quad(Hexahedron(voxel), vertex_set);
		}

		Quad(const Pixel& pixel){
			vertices[0] = pixel[0];
			vertices[1] = pixel[1];
			vertices[2] = pixel[3];
			vertices[3] = pixel[2];
		}

		static const int vtkID = 9;
		std::vector<long unsigned int> getface(const int face) const;
	};
}

#endif