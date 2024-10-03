#include "geometry/assembly.hpp"

double sgn(double x){
	if (x < 0.0){
		return -1.0;
	}
	return 1.0;
}

namespace GeoVox::geometry{
	void AssemblyNode::push_data_to_children(){
		//push data to children and clear current data
		for (long unsigned int d_idx=0; d_idx<_data.size(); d_idx++){
			for (int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->data_valid(_data[d_idx])){
					_children[c_idx]->_data.push_back(_data[d_idx]);

					//update nvert
					int temp_nvert = 0;
					for (int v_idx=0; v_idx<8; v_idx++){
						if (_data[d_idx].contains(_children[c_idx]->box[v_idx])){
							temp_nvert += 1;
						}
					}
					_children[c_idx]->_nvert = std::max(_children[c_idx]->_nvert, temp_nvert);
				}
			}
		}

		//set current _nvert to be invalid
		_nvert = -1;
	}

	bool AssemblyNode::data_valid(const SuperEllipsoid& P) const{
		//particles are convex. If the entire region is contained in a single particle, no need to add more.
		if (_nvert == 8){
			return false;
		}

		//coarse check if particle collides with child
		if (!GeoVox::geometry::GJK(box, P.bbox())){
			return false;
		}

		//fine check if particle collides with child
		if (!GeoVox::geometry::GJK(box, P)){
			return false;
		}

		return true;
	}


	bool AssemblyNode::in_particle(const Point3& point) const{
		if (_isdivided){
			for(int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->box.contains(point)){
					return _children[c_idx]->in_particle(point);
				}
			}
		}

		if (_nvert==8){
			return true;
		}

		for (long unsigned int d_idx=0; d_idx<_data.size(); d_idx++){
			if (_data[d_idx].contains(point)){
				return true;
			}
		}
		
		return false;
	}


	void AssemblyNode::divide(const int n_divisions){
		if (n_divisions <= 0){
			return;
		}

		//TRAVERSE TO LEAF
		if (_isdivided){
			for (int c_idx=0; c_idx<8; c_idx++){
				_children[c_idx]->divide(n_divisions);
			}
			return;
		}

		//ONLY DIVIDE IF THERE ARE PARTICLES MULTIPLE
		if (_data.size() <= MIN_ASSEMBLY_NUMBER_OF_PARTICLES){
			return;
		}

		//PARTICLES ARE CONVEX, ALL CHILDREN WOULD HAVE _nvert=8
		if (_nvert == 8){
			return;
		}

		//DIVIDE
		make_children();

		//CONTINUE TO CHILDERN
		for (int c_idx=0; c_idx<8; c_idx++){
			_children[c_idx]->divide(n_divisions-1);
		}
	}

	void AssemblyNode::divide(){
		// std::cout << "Node(" << ID << "): _data.size=" << _data.size() << " _depth=" << _depth << std::endl;

		//TRAVERSE TO LEAF
		if (_isdivided){
			for (int c_idx=0; c_idx<8; c_idx++){
				_children[c_idx]->divide();
			}
			return;
		}

		//PARTICLES ARE CONVEX, ALL CHILDREN WOULD HAVE _nvert=8
		if (_nvert == 8){
			return;
		}

		//ONLY DIVIDE IF THERE ARE PARTICLES MULTIPLE (unnecessary?)
		if (_data.size() <= MIN_ASSEMBLY_NUMBER_OF_PARTICLES){
			return;
		}

		//DIVIDE
		if (_data.size()>=_root->max_data_per_leaf){
			make_children();
			for (int c_idx=0; c_idx<8; c_idx++){
				_children[c_idx]->divide();
			}
		}
	}

	// bool AssemblyNode::is_gradiated(){
	// 	if (_isdivided){
	// 		bool result = true;
	// 		for (int c_idx=0; c_idx<8; c_idx++){
	// 			result = result && _children[c_idx]->is_gradiated();
	// 		}
	// 		return result;
	// 	}

	// 	bool result = true;
	// 	Point3 H = box.high() - box.low();
	// 	Point3 neighbor_center;
	// 	AssemblyNode* neighbor;

	// 	neighbor_center = box.center() + Point3(-H[0], 0, 0);
	// 	neighbor = _root->findleaf(neighbor_center);
	// 	if (neighbor!=NULL){
	// 		if (neighbor->_depth > this->_depth+1){
	// 			this->make_children();
	// 			result = this->is_gradiated();
	// 		}else if (this->_depth > neighbor->_depth+1){
	// 			neighbor->make_children();
	// 			result = neighbor->is_gradiated();
	// 		}
	// 	}

	// 	neighbor_center = box.center() + Point3(H[0], 0, 0);
	// 	neighbor = _root->findleaf(neighbor_center);
	// 	if (neighbor!=NULL){
	// 		if (neighbor->_depth > this->_depth+1){
	// 			this->make_children();
	// 			result = this->is_gradiated();
	// 		}else if (this->_depth > neighbor->_depth+1){
	// 			neighbor->make_children();
	// 			result = neighbor->is_gradiated();
	// 		}
	// 	}


	// 	neighbor_center = box.center() + Point3(0, -H[1], 0);
	// 	neighbor = _root->findleaf(neighbor_center);
	// 	if (neighbor!=NULL){
	// 		if (neighbor->_depth > this->_depth+1){
	// 			this->make_children();
	// 			result = this->is_gradiated();
	// 		}else if (this->_depth > neighbor->_depth+1){
	// 			neighbor->make_children();
	// 			result = neighbor->is_gradiated();
	// 		}
	// 	}


	// 	neighbor_center = box.center() + Point3(0, H[1], 0);
	// 	neighbor = _root->findleaf(neighbor_center);
	// 	if (neighbor!=NULL){
	// 		if (neighbor->_depth > this->_depth+1){
	// 			this->make_children();
	// 			result = this->is_gradiated();
	// 		}else if (this->_depth > neighbor->_depth+1){
	// 			neighbor->make_children();
	// 			result = neighbor->is_gradiated();
	// 		}
	// 	}


	// 	neighbor_center = box.center() + Point3(0, 0, -H[2]);
	// 	neighbor = _root->findleaf(neighbor_center);
	// 	if (neighbor!=NULL){
	// 		if (neighbor->_depth > this->_depth+1){
	// 			this->make_children();
	// 			result = this->is_gradiated();
	// 		}else if (this->_depth > neighbor->_depth+1){
	// 			neighbor->make_children();
	// 			result = neighbor->is_gradiated();
	// 		}
	// 	}


	// 	neighbor_center = box.center() + Point3(0, 0, H[2]);
	// 	neighbor = _root->findleaf(neighbor_center);
	// 	if (neighbor!=NULL){
	// 		if (neighbor->_depth > this->_depth+1){
	// 			this->make_children();
	// 			result = this->is_gradiated();
	// 		}else if (this->_depth > neighbor->_depth+1){
	// 			neighbor->make_children();
	// 			result = neighbor->is_gradiated();
	// 		}
	// 	}

	// 	return result;
	// }


	


	// void AssemblyNode::makeElements(const std::map<long unsigned int, long unsigned int>& reduced_index, std::vector<std::vector<long unsigned int>> &elem2node, std::vector<int> &elemMarkers) const{
	// 	if (_isdivided){
	// 		for (int i=0; i<8; i++){
	// 			_children[i]->makeElements(reduced_index, elem2node, elemMarkers);
	// 		}
	// 	}else{
	// 		//set nodes
	// 		// unsigned int vtkID = 11; //voxel default
	// 		// bool used_nodes[26] {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

	// 		// //check if this element is non-conforming (ASSUMES MESH IS GRADATED)
	// 		// Point3 H = box.high() - box.low();
	// 		// Point3 neighbor_center;
	// 		// Node* neighbor;

	// 		// neighbor_center = box.center() + Point3(-H[0], 0, 0);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// if (neighbor!=NULL){
	// 		// 	if (neighbor->_isdivided){
	// 		// 		used_nodes[20] = 1;
	// 		// 		used_nodes[11] = 1;
	// 		// 		used_nodes[15] = 1;
	// 		// 		used_nodes[16] = 1;
	// 		// 		used_nodes[19] = 1;
	// 		// 		vtkID = 41; //OctCell (Convex Cell)
	// 		// 	}
	// 		// }

	// 		// neighbor_center = box.center() + Point3(H[0], 0, 0);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// if (neighbor!=NULL){
	// 		// 		if (neighbor->_isdivided){
	// 		// 		used_nodes[21] = 1;
	// 		// 		used_nodes[9]  = 1;
	// 		// 		used_nodes[13] = 1;
	// 		// 		used_nodes[17] = 1;
	// 		// 		used_nodes[18] = 1;
	// 		// 		vtkID = 41; //OctCell (Convex Cell)
	// 		// 	}
	// 		// }

	// 		// neighbor_center = box.center() + Point3(0, -H[1], 0);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// if (neighbor!=NULL){
	// 		// 	if (neighbor->_isdivided){
	// 		// 		used_nodes[22] = 1;
	// 		// 		used_nodes[8]  = 1;
	// 		// 		used_nodes[12] = 1;
	// 		// 		used_nodes[16] = 1;
	// 		// 		used_nodes[17] = 1;
	// 		// 		vtkID = 41; //OctCell (Convex Cell)
	// 		// 	}
	// 		// }

	// 		// neighbor_center = box.center() + Point3(0, H[1], 0);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// if (neighbor!=NULL){
	// 		// 	if (neighbor->_isdivided){
	// 		// 		used_nodes[21] = 1;
	// 		// 		used_nodes[9]  = 1;
	// 		// 		used_nodes[13] = 1;
	// 		// 		used_nodes[17] = 1;
	// 		// 		used_nodes[18] = 1;
	// 		// 		vtkID = 41; //OctCell (Convex Cell)
	// 		// 	}
	// 		// }

	// 		// neighbor_center = box.center() + Point3(0, 0, -H[2]);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// if (neighbor!=NULL){
	// 		// 	if (neighbor->_isdivided){
	// 		// 		used_nodes[24] = 1;
	// 		// 		used_nodes[8]  = 1;
	// 		// 		used_nodes[9]  = 1;
	// 		// 		used_nodes[10] = 1;
	// 		// 		used_nodes[11] = 1;
	// 		// 		vtkID = 41; //OctCell (Convex Cell)
	// 		// 	}
	// 		// }

	// 		// neighbor_center = box.center() + Point3(0, 0, H[2]);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// neighbor = _root->findnode(_depth, neighbor_center);
	// 		// if (neighbor!=NULL){
	// 		// 	if (neighbor->_isdivided){
	// 		// 		used_nodes[25] = 1;
	// 		// 		used_nodes[12] = 1;
	// 		// 		used_nodes[13] = 1;
	// 		// 		used_nodes[14] = 1;
	// 		// 		used_nodes[15] = 1;
	// 		// 		vtkID = 41; //OctCell (Convex Cell)
	// 		// 	}
	// 		// }









	// 		long unsigned int global_index[8];
	// 		get_global_vertex_index(global_index);

	// 		elem2node.push_back({reduced_index.at(global_index[0]),
	// 							reduced_index.at(global_index[1]), 
	// 							reduced_index.at(global_index[2]), 
	// 							reduced_index.at(global_index[3]),
	// 							reduced_index.at(global_index[4]),
	// 							reduced_index.at(global_index[5]),
	// 							reduced_index.at(global_index[6]),
	// 							reduced_index.at(global_index[7])});

	// 		// int nvert = 0;
	// 		// for (int v_idx=0; v_idx<8; v_idx++){
	// 		// 	if (in_particle(box[v_idx])){
	// 		// 		nvert+=1;
	// 		// 	}
	// 		// }
	// 		// elemMarkers.push_back(_data.size());
	// 		elemMarkers.push_back(nvert());
	// 	}
	// }

	// void Assembly::gradiate(){
	// 	if (!_isdivided){
	// 		return;
	// 	}

	// 	for (unsigned int i=0; i<_maxdepth+1; i++){ //should finish on loop 0
	// 		if (is_gradiated()){
	// 			break;
	// 		}
	// 	}
	// }


	//ASSEMBLY
	void Assembly::readfile(const std::string fullfile){
		std::ifstream _file(fullfile);
		std::string line;

		if( not _file.is_open() )
		{
			std::cout << "Could not open " << fullfile << std::endl;
			return;
		}

		//READ FILE
		double rx, ry, rz, eps1, eps2, x, y, z, q0, q1, q2, q3;

		long unsigned int particle_number = 0;

		while (getline(_file, line)){
			if (!line.empty() && line[0] != '#'){
				std::istringstream iss(line);
				iss >> rx;
				iss >> ry;
				iss >> rz;
				iss >> eps1;
				iss >> eps2;
				iss >> x;
				iss >> y;
				iss >> z;
				iss >> q0;
				iss >> q1;
				iss >> q2;
				iss >> q3;

				// std::cout << rx << "\t" << ry << "\t"<< rz << "\t"<< eps1 << "\t"<< eps2 << "\t"<< x << "\t" << y << "\t" << z << "\t" << q0 << "\t"<< q1 << "\t" << q2 << "\t" << q3 << "\n";
				SuperEllipsoid P = SuperEllipsoid(Point3(rx,ry,rz), eps1, eps2, Point3(x,y,z), Quaternion(q0,-q1,-q2,-q3));
				_particles.push_back(P);
				// _particle_index.push_back(particle_number);
				particle_number += 1;
			}
		}

		_setbbox();

	}

	void Assembly::print(std::ostream &stream) const{
		stream << "#For superellipsoids (12 columns:rx	ry	rz	eps1	eps2	x	y	z	q0	q1	q2	q3)\n";
		stream << std::scientific; //set format

		for (long unsigned int i=0; i<_particles.size(); i++){
			//get properties from particle
			SuperEllipsoid particle = _particles[i];
			Point3 r = particle.radius();
			double eps1 = particle.eps1();
			double eps2 = particle.eps2();
			Point3 center = particle.center();
			GeoVox::util::Quaternion Q = particle.quaternion();

			//print to stream
			stream << r[0] << " " << r[1] << " "<< r[2] << " "<< eps1 << " "<< eps2 << " "<< center[0] << " " << center[1] << " " << center[2] << " " << Q[0] << " "<< Q[1] << " " << Q[2] << " " << Q[3] << "\n";
		}
	}




	void Assembly::save_geometry(const std::string filename, const Box& box, const long unsigned int N[3]) const{
		//////////////// OPEN FILE ////////////////
		std::ofstream geofile(filename);

		if (not geofile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			geofile.close();
			return;
		}


		//////////////// WRITE FILE ////////////////
		std::stringstream buffer;

		//HEADER
		buffer << "nx= " << N[0] << std::endl;
		buffer << "ny= " << N[1] << std::endl;
		buffer << "nz= " << N[2] << std::endl;


		//DATA
		Point3 centroid = Point3(0,0,0);
		Point3 H = box.high()-box.low();
		H[0]/=N[0];
		H[1]/=N[1];
		H[2]/=N[2];

		for (long unsigned int  k=0; k<N[2]; k++){
			centroid[2] = box.low()[2] + H[2]*(0.5+k);
			for (long unsigned int  j=0; j<N[1]; j++){
				centroid[1] = box.low()[1] + H[1]*(0.5+j);
				for (long unsigned int  i=0; i<N[0]; i++){
					centroid[0] = box.low()[0] + H[0]*(0.5+i);
					buffer << in_particle(centroid) << " ";
				}
				buffer << std::endl;
			}
			buffer << std::endl;

			geofile << buffer.rdbuf();
			buffer.str("");
		}


		//////////////// CLOSE FILE ////////////////
		geofile.close();
	}

	StructuredPoints Assembly::make_structured_mesh(const Box& subbox, const long unsigned int N[3]) const{
		StructuredPoints mesh(subbox, N);
		mesh.pointMarkers.reserve(N[0]*N[1]*N[2]);

		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					mesh.pointMarkers.push_back(this->in_particle(mesh.vertex(i,j,k)));
				}
			}
		}
		return mesh;
	}

	StructuredPoints Assembly::make_structured_mesh(const long unsigned int N[3]) const{
		Point3 H = (box.high()-box.low())/Point3(N[0],N[1],N[2]);
		Box subbox = Box(box.low()+0.5*H, box.high()-0.5*H);
		return make_structured_mesh(subbox, N);
	}



	void Assembly::_setbbox() {
		if (_particles.size() == 0){
			box = Box(Point3(0,0,0), Point3(1,1,1));
		}
		else{
			box = _particles[0].axis_alligned_bbox();
			for (unsigned int i=0; i<_particles.size(); i++){
				box.combine(_particles[i].axis_alligned_bbox());
			}
		}
	}
}