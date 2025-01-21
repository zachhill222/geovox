#include "geometry/voxel_particle_geometry.hpp"

namespace GeoVox::geometry{
	void VoxelParticleGeometry::compute_connectivity(){
		//set markers that are not SOLID_PHASE_MARKER to UNDEFINED_MARKER 
		initialize();

		std::vector<long unsigned int> active_index;

		////////////// FLOW-CONNECTED (POSITIVE MARKER) REGIONS /////////////
		bool found_unmarked_region = true;
		int mkr = 1;

		while (found_unmarked_region){
			//set initial marker and update active_index with new seeds
			found_unmarked_region = find_unmarked_boundary_voxel(active_index, 16);
			for (long unsigned int idx=0; idx<active_index.size(); idx++){
				markers[active_index[idx]] = mkr;
				mkr += 1;
			}

			//spread regions from new seeds
			long unsigned int n_spread = 1;
			while (n_spread > 0){
				n_spread = spread(active_index);
				// std::cout << "connected fluid phase n_spread= " << n_spread << std::endl;
			}
		}
		
		////////////// FLOW-DISCONNECTED (NEGATIVE MARKER) REGIONS /////////////
		found_unmarked_region = true;
		mkr = -1;

		while (found_unmarked_region){
			//set initial marker and update active_index with new seeds
			found_unmarked_region = find_unmarked_voxel(active_index, 16);
			for (long unsigned int idx=0; idx<active_index.size(); idx++){
				markers[active_index[idx]] = mkr;
				mkr -= 1;
			}
			
			//spread markers
			long unsigned int n_spread = 1;
			while (n_spread>0){
				n_spread = spread(active_index);
				// std::cout << "disconnected fluid phase n_spread= " << n_spread << std::endl;
			}
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

			//get linear index for neighbors
			long unsigned int EAST, WEST, NORTH, SOUTH, TOP, BOTTOM;
			EAST = east(i,j,k);
			WEST = west(i,j,k);
			NORTH = north(i,j,k);
			SOUTH = south(i,j,k);
			TOP = top(i,j,k);
			BOTTOM = bottom(i,j,k);

			if (EAST!=current and markers[EAST]!=SOLID_PHASE_MARKER){
				if (markers[EAST]==UNDEFINED_MARKER){
					markers[EAST]=markers[current];
					new_active_index.push_back(EAST);
					n_spread += 1;
				}else if (markers[EAST]!=markers[current]){ //check if we should merge
					int mkr_low = std::min(markers[EAST], markers[current]);
					int mkr_high = std::max(markers[EAST], markers[current]);
					merge_regions(mkr_low, mkr_high);
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
					merge_regions(mkr_low, mkr_high);
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
					merge_regions(mkr_low, mkr_high);
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
					merge_regions(mkr_low, mkr_high);
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
					merge_regions(mkr_low, mkr_high);
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
					merge_regions(mkr_low, mkr_high);
				}
			}
			
		}

		//update active_index and return
		active_index = new_active_index;
		return n_spread;
	}

	void VoxelParticleGeometry::merge_regions(const int mkr_low, const int mkr_high){
		int old_mkr, new_mkr;
		if (mkr_low>0){//if lower of the two markers is positive, use that as the region marker
			old_mkr = mkr_high;
			new_mkr = mkr_low;
		}else{
			old_mkr = mkr_low;
			new_mkr = mkr_high;
		}

		// std::cout << "\tMERGE " << old_mkr << " <- " << new_mkr << std::endl;
		replace_marker(old_mkr, new_mkr);
	}

	void VoxelParticleGeometry::initialize(){
		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					if (markers[index(i,j,k)] != SOLID_PHASE_MARKER){
						markers[index(i,j,k)] = UNDEFINED_MARKER;
					}
				}
			}
		}
	}

	


	bool VoxelParticleGeometry::find_unmarked_boundary_voxel(std::vector<long unsigned int> &active_index, const long unsigned int max_voxels) const {
		active_index.clear();

		//search x-faces
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				if (not wall_bc[0]){
					if (markers[index(0,j,k)] == UNDEFINED_MARKER){
						active_index.push_back(index(0,j,k));
						if (active_index.size()>=max_voxels){return true;}
					}
				}
				if (not wall_bc[1]){
					if (markers[index(N[0]-1,j,k)] == UNDEFINED_MARKER){
						active_index.push_back(index(N[0]-1,j,k));
						if (active_index.size()>=max_voxels){return true;}
					}
				}
			}
		}

		//search y-faces
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int i=0; i<N[0]; i++){
				if (not wall_bc[2]){
					if (markers[index(i,0,k)] == UNDEFINED_MARKER){
						active_index.push_back(index(i,0,k));
						if (active_index.size()>=max_voxels){return true;}
					}
				}
				if (not wall_bc[3]){
					if (markers[index(i,N[1]-1,k)] == UNDEFINED_MARKER){
						active_index.push_back(index(i,N[1]-1,k));
						if (active_index.size()>=max_voxels){return true;}
					}
				}
			}
		}

		//search z-faces
		for (long unsigned int j=0; j<N[1]; j++){
			for (long unsigned int i=0; i<N[0]; i++){
				if (not wall_bc[4]){
					if (markers[index(i,j,0)] == UNDEFINED_MARKER){
						active_index.push_back(index(i,j,0));
						if (active_index.size()>=max_voxels){return true;}
					}
				}
				if (not wall_bc[5]){
					if (markers[index(i,j,N[2]-1)] == UNDEFINED_MARKER){
						active_index.push_back(index(i,j,N[2]-1));
						if (active_index.size()>=max_voxels){return true;}
					}
				}
			}
		}

		return active_index.size() > 0;
	}


	bool VoxelParticleGeometry::find_unmarked_voxel(std::vector<long unsigned int> &active_index, const long unsigned int max_voxels) const {
		active_index.clear();

		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					if (markers[index(i,j,k)] == UNDEFINED_MARKER){
						active_index.push_back(index(i,j,k));
						if (active_index.size()>=max_voxels){return true;}
					}
				}
			}
		}

		return active_index.size() > 0;
	}

	void VoxelParticleGeometry::print(std::ostream &stream) const{
		std::vector<int> mkr;
		std::vector<long unsigned int> mkr_count;
		unique_markers(mkr, mkr_count);


		double pos_count=0;
		double neg_count=0;
		double zero_count=0;

		for (long unsigned int idx=0; idx<mkr.size(); idx++){
			mkr_count[idx] = count(mkr[idx]);
			if (mkr[idx]<0){
				neg_count += mkr_count[idx];
			}else if (mkr[idx]>0){
				pos_count += mkr_count[idx];
			}else{
				zero_count += mkr_count[idx];
			}
		}


		for (long unsigned int idx=0; idx<mkr.size(); idx++){
			if (mkr_count[idx]>=5){
				stream << "marker= " << mkr[idx] << "\tcount= " << mkr_count[idx] << "\tfraction= " << static_cast<double>(mkr_count[idx])/(N[0]*N[1]*N[2]) << std::endl;
			}
		}

		stream << "\npositive_marker_count= " << pos_count  << "\tpositive_marker_fraction= " << pos_count/(N[0]*N[1]*N[2])  << std::endl;
		stream << "negative_marker_count= " << neg_count  << "\tnegative_marker_fraction= " << neg_count/(N[0]*N[1]*N[2])  << std::endl;
		stream << "zero_marker_count= "     << zero_count << "\tzero_marker_fraction= "     << zero_count/(N[0]*N[1]*N[2]) << std::endl;
	}

	std::string VoxelParticleGeometry::tostr() const{
		std::stringstream ss;
		print(ss);
		return ss.str();
	}
}