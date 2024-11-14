#include "geometry/voxel_particle_geometry.hpp"

namespace GeoVox::geometry{
	long unsigned int VoxelParticleGeometry::index_bc(long unsigned int i, long unsigned int j, long unsigned int k){
		if (periodicBC[0]){
			i=i%N[0];
		}else{
			i=std::min(i,N[0]-1);
		}

		if (periodicBC[1]){
			j=j%N[1];
		}else{
			j=std::min(j,N[1]-1);
		}

		if (periodicBC[2]){
			k=k%N[2];
		}else{
			k=std::min(k,N[2]-1);
		}

		return index(i,j,k);
	}

	void VoxelParticleGeometry::compute_connectivity(){
		std::vector<long unsigned int> active_index;
		std::set<int> unique_markers;

		////////////// FLOW-CONNECTED (POSITIVE MARKER) REGIONS /////////////
		//set seeds
		for (int n=0; n<6; n++){
			long unsigned int i,j,k;
			if (not wallBC[0]){
				if (find_unmarked_face(0,i,j,k)){
					markers[index(i,j,k)] = n+1;
					unique_markers.insert(n+1);
					active_index.push_back(index(i,j,k));
				}
			}
		}
		std::cout << "set initial seeds\n";
		
		long unsigned int n_spread = 1;
		int iter = 0;
		while (n_spread > 0){
			n_spread = spread(active_index);
			saveas("particle_label_spread_" + std::to_string(++iter) + ".vtk");
		}
	}


	long unsigned int VoxelParticleGeometry::spread(std::vector<long unsigned int> &active_index){
		std::vector<long unsigned int> new_active_index;
		std::set<std::array<int, 2>> merge_markers;

		long unsigned int n_spread = 0;

		#pragma omp parallel for reduction(+:n_spread) //race condition on spread is ok
		for (long unsigned int n=0; n<active_index.size(); n++){
			long unsigned int i, j, k, neighbor;
			const long unsigned int current = active_index[n];

			//get i,j,k indexing of current voxel
			index2ijk(current,i,j,k);


			neighbor = index_bc(i-1,j,k); //at non-periodic BC, we might have neighbor == active_index[n]
			if (neighbor != active_index[n] and markers[neighbor] != SOLID_PHASE_MARKER){
				if (markers[neighbor]==UNDEFINED_MARKER){
					markers[neighbor]=markers[current];
					new_active_index.push_back(neighbor);
					n_spread += 1;

				}else if (markers[neighbor] != markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[neighbor], markers[current]);
					int mkr_high = std::max(markers[neighbor], markers[current]);

					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}
			
			neighbor = index_bc(i+1,j,k); //at non-periodic BC, we might have neighbor == active_index[n]
			if (neighbor != active_index[n] and markers[neighbor] != SOLID_PHASE_MARKER){
				if (markers[neighbor]==UNDEFINED_MARKER){
					markers[neighbor]=markers[current];
					new_active_index.push_back(neighbor);
					n_spread += 1;

				}else if (markers[neighbor] != markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[neighbor], markers[current]);
					int mkr_high = std::max(markers[neighbor], markers[current]);

					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}

			neighbor = index_bc(i,j-1,k); //at non-periodic BC, we might have neighbor == active_index[n]
			if (neighbor != active_index[n] and markers[neighbor] != SOLID_PHASE_MARKER){
				if (markers[neighbor]==UNDEFINED_MARKER){
					markers[neighbor]=markers[current];
					new_active_index.push_back(neighbor);
					n_spread += 1;

				}else if (markers[neighbor] != markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[neighbor], markers[current]);
					int mkr_high = std::max(markers[neighbor], markers[current]);

					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}

			neighbor = index_bc(i,j+1,k); //at non-periodic BC, we might have neighbor == active_index[n]
			if (neighbor != active_index[n] and markers[neighbor] != SOLID_PHASE_MARKER){
				if (markers[neighbor]==UNDEFINED_MARKER){
					markers[neighbor]=markers[current];
					new_active_index.push_back(neighbor);
					n_spread += 1;

				}else if (markers[neighbor] != markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[neighbor], markers[current]);
					int mkr_high = std::max(markers[neighbor], markers[current]);

					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}

			neighbor = index_bc(i,j,k-1); //at non-periodic BC, we might have neighbor == active_index[n]
			if (neighbor != active_index[n] and markers[neighbor] != SOLID_PHASE_MARKER){
				if (markers[neighbor]==UNDEFINED_MARKER){
					markers[neighbor]=markers[current];
					new_active_index.push_back(neighbor);
					n_spread += 1;

				}else if (markers[neighbor] != markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[neighbor], markers[current]);
					int mkr_high = std::max(markers[neighbor], markers[current]);

					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}

			neighbor = index_bc(i,j,k+1); //at non-periodic BC, we might have neighbor == active_index[n]
			if (neighbor != active_index[n] and markers[neighbor] != SOLID_PHASE_MARKER){
				if (markers[neighbor]==UNDEFINED_MARKER){
					markers[neighbor]=markers[current];
					new_active_index.push_back(neighbor);
					n_spread += 1;

				}else if (markers[neighbor] != markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[neighbor], markers[current]);
					int mkr_high = std::max(markers[neighbor], markers[current]);

					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}
		} //end par-for

		//merge regions
		for (auto low_high : merge_markers){
			int old_mkr, new_mkr;
			if (low_high[0]>0){//if lower of the two markers is positive, use that as the region marker
				old_mkr = low_high[1];
				new_mkr = low_high[0];
			}else{
				old_mkr = low_high[0];
				new_mkr = low_high[1];
			}
			replace_marker(old_mkr, new_mkr);
		}

		//update active_index and return
		active_index = new_active_index;
		return n_spread;
	}


	void VoxelParticleGeometry::initialize(){
		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					if (A->in_particle(idx2point(i,j,k))){
						markers[index(i,j,k)] = SOLID_PHASE_MARKER;
					}else{
						markers[index(i,j,k)] = UNDEFINED_MARKER;
					}
				}
			}
		}
	}

	bool VoxelParticleGeometry::find_unmarked_face(const int n, long unsigned int &ii, long unsigned int &jj, long unsigned int &kk){
		switch(n) {
		case 0: //xlow
			ii=0;
			for (long unsigned int k=0; k<N[2]; k++){
				for (long unsigned int j=0; j<N[1]; j++){
					if (markers[index(ii,j,k)==UNDEFINED_MARKER]){
						jj=j;
						kk=k;
						return true;
					}
				}
			}
			break;

		case 1: //xhigh
			ii=N[0]-1;
			for (long unsigned int k=0; k<N[2]; k++){
				for (long unsigned int j=0; j<N[1]; j++){
					if (markers[index(ii,j,k)==UNDEFINED_MARKER]){
						jj=j;
						kk=k;
						return true;
					}
				}
			}
			break;

		case 2: //ylow
			jj=0;
			for (long unsigned int k=0; k<N[2]; k++){
				for (long unsigned int i=0; i<N[0]; i++){
					if (markers[index(i,jj,k)==UNDEFINED_MARKER]){
						ii=i;
						kk=k;
						return true;
					}
				}
			}
			break;

		case 3: //yhigh
			jj=N[1]-1;
			for (long unsigned int k=0; k<N[2]; k++){
				for (long unsigned int i=0; i<N[0]; i++){
					if (markers[index(i,jj,k)==UNDEFINED_MARKER]){
						ii=i;
						kk=k;
						return true;
					}
				}
			}
			break;

		case 4: //zlow
			kk=0;
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					if (markers[index(i,j,kk)==UNDEFINED_MARKER]){
						ii=i;
						jj=j;
						return true;
					}
				}
			}
			break;

		case 5: //zhigh
			kk=N[2]-1;
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					if (markers[index(i,j,kk)==UNDEFINED_MARKER]){
						ii=i;
						jj=j;
						return true;
					}
				}
			}
			break;
		}

		return false;
	}
}