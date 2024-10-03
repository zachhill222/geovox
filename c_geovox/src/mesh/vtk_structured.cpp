#include "mesh/vtk_structured.hpp"


namespace GeoVox::mesh{
	void StructuredPoints::saveas(const std::string filename) const{
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
		buffer << "DIMENSIONS " << N[0] << " " << N[1] << " " << N[2] << "\n";
		buffer << "ORIGIN " << box.low().tostring() << "\n";
		buffer << "SPACING " << H.tostring() << "\n\n";

		meshfile << buffer.rdbuf();
		buffer.str("");


		//ELEMENT_MARKERS
		buffer << "POINT_DATA " << N[0]*N[1]*N[2] << "\n";
		buffer << "SCALARS pointMarkers integer\n";
		buffer << "LOOKUP_TABLE default\n";
		for (long unsigned int k=0; k<N[2]; k++){
			long unsigned int start_idx = N[0]*N[1]*k;
			for (long unsigned int ij=0; ij<N[0]*N[1]; ij++){
				buffer << pointMarkers[start_idx+ij] << " ";
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

		H = ((box.high()-box.low())/Point3(N[0], N[1], N[2]));

		//READ BODY
		int mkr;
		pointMarkers.reserve(N[0]*N[1]*N[2]);

		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					geofile >> mkr;
					pointMarkers.push_back(mkr);
				}
			}
		}
		geofile.close();
	}
}