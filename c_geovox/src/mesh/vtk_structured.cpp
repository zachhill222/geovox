#include "mesh/vtk_structured.hpp"


namespace GeoVox::mesh{
	Point3 StructuredPoints::idx2point(long unsigned int i, long unsigned int j, long unsigned int k) const{
		Point3 low = box.low();
		double ii = static_cast<double>(i);
		double jj = static_cast<double>(j);
		double kk = static_cast<double>(k);

		switch (dof_location){
		case 0: return low + Point3(H[0]*ii,       H[1]*(jj+0.5), H[2]*(kk+0.5));
		case 1: return low + Point3(H[0]*(ii+0.5), H[1]*jj,       H[2]*(kk+0.5));
		case 2: return low + Point3(H[0]*(ii+0.5), H[1]*(jj+0.5), H[2]*kk      );
		case 3: return low + Point3(H[0]*(ii+0.5), H[1]*(jj+0.5), H[2]*(kk+0.5));
		default: throw std::runtime_error("Unknown DOF location. Must be between 0 and 3. (x-face, y-face, z-face, centroid).");
		}

		return Point3(0,0,0);
	}

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



	long unsigned int StructuredPoints::east(long unsigned int i, long unsigned int j, long unsigned int k) const{
		if (i+1<N[0]){
			return index(i+1,j,k);
		}

		if (periodic_bc[0]){
			return index(0,j,k);
		}else{
			return index(i,j,k);
		}
	}

	long unsigned int StructuredPoints::north(long unsigned int i, long unsigned int j, long unsigned int k) const{
		if (j+1<N[1]){
			return index(i,j+1,k);
		}

		if (periodic_bc[1]){
			return index(i,0,k);
		}else{
			return index(i,j,k);
		}
	}

	long unsigned int StructuredPoints::top(long unsigned int i, long unsigned int j, long unsigned int k) const{
		if (k+1<N[2]){
			return index(i,j,k+1);
		}

		if (periodic_bc[2]){
			return index(i,j,0);
		}else{
			return index(i,j,k);
		}
	}

	long unsigned int StructuredPoints::west(long unsigned int i, long unsigned int j, long unsigned int k) const{
		if (i){ //i>0
			return index(i-1,j,k);
		}

		if (periodic_bc[0]){
			return index(N[0]-1,j,k);
		}else{
			return index(0,j,k);
		}
	}

	long unsigned int StructuredPoints::south(long unsigned int i, long unsigned int j, long unsigned int k) const{
		if (j){ //j>0
			return index(i,j-1,k);
		}

		if (periodic_bc[1]){
			return index(i,N[1]-1,k);
		}else{
			return index(i,0,k);
		}
	}

	long unsigned int StructuredPoints::bottom(long unsigned int i, long unsigned int j, long unsigned int k) const{
		if (k){ //k>0
			return index(i,j,k-1);
		}

		if (periodic_bc[2]){
			return index(i,j,N[2]-1);
		}else{
			return index(i,j,0);
		}
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



	long unsigned int StructuredPoints::count(const int mkr) const{
		long unsigned int number = 0;
		#pragma omp parallel for collapse(3) reduction(+:number)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					long unsigned int idx = index(i,j,k);
					if (markers[idx] == mkr){
						number += 1;
					}
				}
			}
		}

		return number;
	}


	
		void StructuredPoints::unique_markers(std::vector<int> &mkr, std::vector<long unsigned int> &mkr_count) const{
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					long unsigned int idx = index(i,j,k);

					long unsigned int mkr_idx = std::distance(mkr.begin(), std::find(mkr.begin(), mkr.end(), markers[idx]));
					if (mkr_idx==mkr.size()){
						mkr.push_back(markers[idx]);
						mkr_count.push_back(1);
					}else{
						mkr_count[mkr_idx] += 1;
					}
				}
			}
		}
	}

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
		switch (dof_location){
		case 0:
			buffer << "DIMENSIONS " << N[0] << " " << N[1] << " " << N[2] << "\n";
			buffer << "ORIGIN " << box.low()+Point3(0,0.5*H[1],0.5*H[2]) << "\n";
		case 1:
			buffer << "DIMENSIONS " << N[0] << " " << N[1] << " " << N[2] << "\n";
			buffer << "ORIGIN " << box.low()+Point3(0.5*H[0],0,0.5*H[2]) << "\n";
		case 2:
			buffer << "DIMENSIONS " << N[0] << " " << N[1] << " " << N[2] << "\n";
			buffer << "ORIGIN " << box.low()+Point3(0.5*H[0],0.5*H[1],0) << "\n";
		case 3:
			buffer << "DIMENSIONS " << N[0]+1 << " " << N[1]+1 << " " << N[2]+1 << "\n";
			buffer << "ORIGIN " << box.low() << "\n";
		}


		buffer << "SPACING " << H << "\n\n";

		meshfile << buffer.rdbuf();
		buffer.str("");


		//POINT_MARKERS (CENTROIDS OF CELLS)
		if (dof_location==3){
			buffer << "CELL_DATA " << N[0]*N[1]*N[2] << "\n";
		}else{
			buffer << "POINT_DATA " << N[0]*N[1]*N[2] << "\n";
		}
		buffer << "SCALARS markers integer\n";
		buffer << "LOOKUP_TABLE default\n";
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					buffer << markers[index(i,j,k)] << " ";
				}
			}
			buffer << "\n";
		}
		buffer << "\n";
		
		meshfile << buffer.rdbuf();
		buffer.str("");

		//////////////// CLOSE FILE ////////////////
		meshfile.close();
	}

	// void StructuredPoints::readfile(const std::string filename){
	// 	//OPEN FILE
	// 	std::ifstream geofile(filename);
	// 	std::string str;

	// 	if (not geofile.is_open()){
	// 		std::cout << "Could not open " << filename << std::endl;
	// 		return;
	// 	}


	// 	//READ HEADER
	// 	geofile >> str >> N[0];
	// 	geofile >> str >> N[1];
	// 	geofile >> str >> N[2];

	// 	H = (box.high()-box.low()).array()/Point3(N[0], N[1], N[2]).array();

	// 	//READ BODY
	// 	int mkr;
	// 	markers.reserve(N[0]*N[1]*N[2]);

	// 	for (long unsigned int k=0; k<N[2]; k++){
	// 		for (long unsigned int j=0; j<N[1]; j++){
	// 			for (long unsigned int i=0; i<N[0]; i++){
	// 				geofile >> mkr;
	// 				markers.push_back(mkr);
	// 			}
	// 		}
	// 	}
	// 	geofile.close();
	// }
}