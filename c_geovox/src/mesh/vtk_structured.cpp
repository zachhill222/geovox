#include "mesh/vtk_structured.hpp"


namespace GeoVox::mesh{
	bool StructuredPoints::index2ijk(long unsigned int l, long unsigned int &i, long unsigned int &j, long unsigned int &k) const{
		if (l >= N[0]*N[1]*N[2]){
			return false;
		}

		i = l%N[0];
		l = (l-i)/N[0];

		j = l%N[1];
		k = (l-j)/N[1];

		return true;
	}


	void StructuredPoints::set_all_markers(const int mkr){
		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					markers[index(i,j,k)] = mkr;
				}
			}
		}
	}

	void StructuredPoints::replace_marker(const int old_mkr, const int new_mkr){
		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					long unsigned int idx = index(i,j,k);
					if (markers[idx] == old_mkr){
						markers[idx] = new_mkr;
					}
				}
			}
		}
	}

	void StructuredPoints::saveas(const std::string filename, bool cells) const{
		//////////////// OPEN FILE ////////////////
		std::ofstream meshfile(filename);

		if (not meshfile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			meshfile.close();
			return;
		}

		//////////////// WRITE TO FILE ////////////////
		std::stringstream buffer;

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << "Mesh Data\n";
		buffer << "ASCII\n\n";

		//POINTS (CENTROIDS)
		buffer << "DATASET STRUCTURED_POINTS\n";
		if (cells){
			buffer << "DIMENSIONS " << N[0]+1 << " " << N[1]+1 << " " << N[2]+1 << "\n";
		}else{
			buffer << "DIMENSIONS " << N[0] << " " << N[1] << " " << N[2] << "\n";
		}
		
		buffer << "ORIGIN " << box.low() << "\n";
		buffer << "SPACING " << H << "\n\n";

		meshfile << buffer.rdbuf();
		buffer.str("");


		//POINT_MARKERS (CENTROIDS OF CELLS)
		if (cells){
			buffer << "CELL_DATA " << N[0]*N[1]*N[2] << "\n";
		}else{
			buffer << "POINT_DATA " << N[0]*N[1]*N[2] << "\n";
		}
		buffer << "SCALARS markers integer\n";
		buffer << "LOOKUP_TABLE default\n";
		for (long unsigned int k=0; k<N[2]; k++){
			long unsigned int start_idx = N[0]*N[1]*k;
			for (long unsigned int ij=0; ij<N[0]*N[1]; ij++){
				buffer << markers[start_idx+ij] << " ";
			}
			buffer << "\n";
		}
		buffer << "\n";
		
		meshfile << buffer.rdbuf();
		buffer.str("");

		//////////////// CLOSE FILE ////////////////
		meshfile.close();
	}

	void StructuredPoints::readfile(const std::string filename){
		//OPEN FILE
		std::ifstream geofile(filename);
		std::string str;

		if (not geofile.is_open()){
			std::cout << "Could not open " << filename << std::endl;
			return;
		}


		//READ HEADER
		geofile >> str >> N[0];
		geofile >> str >> N[1];
		geofile >> str >> N[2];

		H = (box.high()-box.low()).array()/Point3(N[0], N[1], N[2]).array();

		//READ BODY
		int mkr;
		markers.reserve(N[0]*N[1]*N[2]);

		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					geofile >> mkr;
					markers.push_back(mkr);
				}
			}
		}
		geofile.close();
	}
}