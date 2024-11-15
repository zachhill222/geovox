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
				if (find_unmarked_face(n,i,j,k)){
					markers[index(i,j,k)] = n+1;
					unique_markers.insert(n+1);
					active_index.push_back(index(i,j,k));
				}
			}
		}
		std::cout << "set initial seeds\n";
		saveas("particle_label_spread_0.vtk", true);

		long unsigned int n_spread = 1;
		int iter = 0;
		while (n_spread > 0){
			n_spread = spread(active_index);
			
			iter += 1;
			std::cout << "pass " << iter << std::endl;
			std::cout << "\tn_spread= " << n_spread << std::endl;
			std::cout << "\tactive_index.size()= " << active_index.size() << std::endl;


			std::string filename = "particle_label_spread_" + std::to_string(iter) + ".vtk";
			std::cout << "saving: " + filename << std::endl;
			saveas(filename, true);
		}
	}


	long unsigned int VoxelParticleGeometry::spread(std::vector<long unsigned int> &active_index){
		std::vector<long unsigned int> new_active_index;
		std::set<std::array<int, 2>> merge_markers;

		long unsigned int n_spread = 0;

		// #pragma omp parallel for reduction(+:n_spread) //race condition on spread is ok
		for (long unsigned int n=0; n<active_index.size(); n++){
			long unsigned int i, j, k;
			const long unsigned int current = active_index[n];

			//get i,j,k indexing of current voxel
			index2ijk(current,i,j,k);
			// std::cout << "current= " << current << " -> (i,j,k)= (" << i << ", " <<  j << ", " <<  k << ", " << ")\n";
			// std::cout << "index(i,j,k) = " << index(i,j,k) << std::endl;

			//get linear index for neighbors
			long unsigned int EAST, WEST, NORTH, SOUTH, TOP, BOTTOM;
			
			//X (EAST/WEST)
			if (i==0){
				if (periodicBC[0]){
					WEST = index(N[0]-1,j,k);
				}else{
					WEST = current;
				}

				EAST = index(i+1,j,k);
			}else if (i==N[0]-1){
				if (periodicBC[0]){
					EAST = index(0,j,k);
				}else{
					EAST = current;
				}

				WEST = index(i-1,j,k);
			}else{
				EAST = index(i+1,j,k);
				WEST = index(i-1,j,k);
			}

			//Y (NORTH/SOUTH)
			if (j==0){
				if (periodicBC[1]){
					SOUTH = index(i,N[1]-1,k);
				}else{
					SOUTH = current;
				}

				NORTH = index(i,j+1,k);
			}else if (j==N[1]-1){
				if (periodicBC[1]){
					NORTH = index(i,0,k);
				}else{
					NORTH = current;
				}

				SOUTH = index(i,j-1,k);
			}else{
				NORTH = index(i,j+1,k);
				SOUTH = index(i,j-1,k);
			}

			//Z (TOP/BOTTOM)
			if (k==0){
				if (periodicBC[2]){
					BOTTOM = index(i,j,N[2]-1);
				}else{
					BOTTOM = current;
				}

				TOP = index(i,j,k+1);
			}else if (k==N[2]-1){
				if (periodicBC[2]){
					TOP = index(i,j,0);
				}else{
					TOP = current;
				}

				BOTTOM = index(i,j,k-1);
			}else{
				TOP    = index(i,j,k+1);
				BOTTOM = index(i,j,k-1);
			}


			if (EAST!=current and markers[EAST]!=SOLID_PHASE_MARKER){
				if (markers[EAST]==UNDEFINED_MARKER){
					markers[EAST]=markers[current];
					new_active_index.push_back(EAST);
					n_spread += 1;
				}else if (markers[EAST]!=markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[EAST], markers[current]);
					int mkr_high = std::max(markers[EAST], markers[current]);
					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}
			
			if (WEST!=current and markers[WEST]!=SOLID_PHASE_MARKER){
				if (markers[WEST]==UNDEFINED_MARKER){
					markers[WEST]=markers[current];
					new_active_index.push_back(WEST);
					n_spread += 1;
				}else if (markers[WEST]!=markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[WEST], markers[current]);
					int mkr_high = std::max(markers[WEST], markers[current]);
					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}

			if (NORTH!=current and markers[NORTH]!=SOLID_PHASE_MARKER){
				if (markers[NORTH]==UNDEFINED_MARKER){
					markers[NORTH]=markers[current];
					new_active_index.push_back(NORTH);
					n_spread += 1;
				}else if (markers[NORTH]!=markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[NORTH], markers[current]);
					int mkr_high = std::max(markers[NORTH], markers[current]);
					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}

			if (SOUTH!=current and markers[SOUTH]!=SOLID_PHASE_MARKER){
				if (markers[SOUTH]==UNDEFINED_MARKER){
					markers[SOUTH]=markers[current];
					new_active_index.push_back(SOUTH);
					n_spread += 1;
				}else if (markers[SOUTH]!=markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[SOUTH], markers[current]);
					int mkr_high = std::max(markers[SOUTH], markers[current]);
					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}

			if (TOP!=current and markers[TOP]!=SOLID_PHASE_MARKER){
				if (markers[TOP]==UNDEFINED_MARKER){
					markers[TOP]=markers[current];
					new_active_index.push_back(TOP);
					n_spread += 1;
				}else if (markers[TOP]!=markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[TOP], markers[current]);
					int mkr_high = std::max(markers[TOP], markers[current]);
					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}

			if (BOTTOM!=current and markers[BOTTOM]!=SOLID_PHASE_MARKER){
				if (markers[BOTTOM]==UNDEFINED_MARKER){
					markers[BOTTOM]=markers[current];
					new_active_index.push_back(BOTTOM);
					n_spread += 1;
				}else if (markers[BOTTOM]!=markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[BOTTOM], markers[current]);
					int mkr_high = std::max(markers[BOTTOM], markers[current]);
					merge_markers.insert(std::array<int, 2> {mkr_low, mkr_high});
				}
			}
			
		}

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

			std::cout << "\tMERGE " << old_mkr << " <- " << new_mkr << std::endl;
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
					if (markers[index( ii, (N[1]/2+j)%N[1], (N[2]/2+k)%N[2] )]==UNDEFINED_MARKER){
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
					if (markers[index(ii,j,k)]==UNDEFINED_MARKER){
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
					if (markers[index(i,jj,k)]==UNDEFINED_MARKER){
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
					if (markers[index(i,jj,k)]==UNDEFINED_MARKER){
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
					if (markers[index(i,j,kk)]==UNDEFINED_MARKER){
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
					if (markers[index(i,j,kk)]==UNDEFINED_MARKER){
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