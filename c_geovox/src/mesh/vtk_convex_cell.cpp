#include "mesh/vtk_convex_cell.hpp"

namespace GeoVox::mesh{
	int OctCell::total_nodes() const{
		int result = 8; //first 8 nodes MUST be used
		for (int i=8; i<26; i++){
			result += used_nodes[i];
		}

		return result;
	}


	std::vector<long unsigned int> OctCell::getface(const int face) const{
		//vertices are returned in VTK_PIXEL order
		//determine which "large face" and "sub face"
		int large_face;
		int face_counter = 0;
		int subface = 0;

		for (large_face=0; large_face<6; large_face++){
			subface = 0;

			if (used_nodes[20+large_face]){
				for (int i=0; i<4; i++){
					subface += 1;
					face_counter += 1;
					if (face_counter == face){
						break;
					}
				}
			}else{
				face_counter += 1;
			}

			if (face_counter == face){
				break;
			}
		}

		//return face
		switch (large_face){
		case 0: //center node 20:
			switch (subface){
				case 0: //face is not subdivided
					return std::vector<long unsigned int> {vertices[0], vertices[4], vertices[3], vertices[7]};
					break;
				case 1:
					return std::vector<long unsigned int> {vertices[0], vertices[16], vertices[11], vertices[20]};
					break;
				case 2:
					return std::vector<long unsigned int> {vertices[16], vertices[4], vertices[20], vertices[15]};
					break;
				case 3:
					return std::vector<long unsigned int> {vertices[20], vertices[15], vertices[19], vertices[7]};
					break;
				case 4:
					return std::vector<long unsigned int> {vertices[11], vertices[20], vertices[3], vertices[19]};
					break;
				}

			break;
		case 1: //center node 21:
			switch (subface){
				case 0: //face is not subdivided
					return std::vector<long unsigned int> {vertices[5], vertices[1], vertices[6], vertices[2]};
					break;
				case 1:
					return std::vector<long unsigned int> {vertices[5], vertices[17], vertices[13], vertices[21]};
					break;
				case 2:
					return std::vector<long unsigned int> {vertices[13], vertices[21], vertices[6], vertices[18]};
					break;
				case 3:
					return std::vector<long unsigned int> {vertices[21], vertices[9], vertices[18], vertices[2]};
					break;
				case 4:
					return std::vector<long unsigned int> {vertices[17], vertices[1], vertices[21], vertices[9]};
					break;
				}
				
			break;
		case 2: //center node 22
			switch (subface){
				case 0: //face is not subdivided
					return std::vector<long unsigned int> {vertices[4], vertices[0], vertices[5], vertices[1]};
					break;
				case 1:
					return std::vector<long unsigned int> {vertices[4], vertices[16], vertices[12], vertices[22]};
					break;
				case 2:
					return std::vector<long unsigned int> {vertices[12], vertices[22], vertices[5], vertices[17]};
					break;
				case 3:
					return std::vector<long unsigned int> {vertices[22], vertices[8], vertices[17], vertices[1]};
					break;
				case 4:
					return std::vector<long unsigned int> {vertices[16], vertices[0], vertices[22], vertices[8]};
					break;
				}
				
			break;
		case 3: //center node 23
			switch (subface){
				case 0: //face is not subdivided
					return std::vector<long unsigned int> {vertices[6], vertices[2], vertices[7], vertices[3]};
					break;
				case 1:
					return std::vector<long unsigned int> {vertices[6], vertices[18], vertices[14], vertices[23]};
					break;
				case 2:
					return std::vector<long unsigned int> {vertices[14], vertices[23], vertices[7], vertices[19]};
					break;
				case 3:
					return std::vector<long unsigned int> {vertices[23], vertices[10], vertices[19], vertices[3]};
					break;
				case 4:
					return std::vector<long unsigned int> {vertices[18], vertices[2], vertices[23], vertices[10]};
					break;
				}
				
			break;
		case 4: //center node 24
			switch (subface){
				case 0: //face is not subdivided
					return std::vector<long unsigned int> {vertices[1], vertices[0], vertices[2], vertices[3]};
					break;
				case 1:
					return std::vector<long unsigned int> {vertices[1], vertices[8], vertices[9], vertices[24]};
					break;
				case 2:
					return std::vector<long unsigned int> {vertices[8], vertices[0], vertices[24], vertices[11]};
					break;
				case 3:
					return std::vector<long unsigned int> {vertices[24], vertices[11], vertices[10], vertices[3]};
					break;
				case 4:
					return std::vector<long unsigned int> {vertices[9], vertices[24], vertices[2], vertices[10]};
					break;
				}
				
			break;
		case 5: //center node 25
			switch (subface){
				case 0: //face is not subdivided
					return std::vector<long unsigned int> {vertices[4], vertices[5], vertices[7], vertices[6]};
					break;
				case 1:
					return std::vector<long unsigned int> {vertices[4], vertices[12], vertices[15], vertices[25]};
					break;
				case 2:
					return std::vector<long unsigned int> {vertices[12], vertices[5], vertices[25], vertices[13]};
					break;
				case 3:
					return std::vector<long unsigned int> {vertices[25], vertices[13], vertices[14], vertices[6]};
					break;
				case 4:
					return std::vector<long unsigned int> {vertices[15], vertices[25], vertices[7], vertices[14]};
					break;
				}
				
			break;
		}
	return std::vector<long unsigned int> {};
	}
}