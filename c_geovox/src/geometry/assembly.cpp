#include "geometry/assembly.hpp"

double sgn(double x){
	if (x < 0.0){
		return -1.0;
	}
	return 1.0;
}

namespace GeoVox::geometry{
	bool AssemblyNode::data_valid(const SuperEllipsoid& P) const{
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

	// void AssemblyNode::insert_particle(const SuperEllipsoid P){
	// 	// std::cout << "insert_particle\n";
	// 	std::cout << _isdivided << std::endl;
	// 	if (_isdivided){
	// 		// std::cout << "divided\n";
	// 		for (int c_idx=0; c_idx<8; c_idx++){
	// 			//coarse check if particle collides with child
	// 			if (!GeoVox::geometry::GJK(_children[c_idx]->box, P.bbox())){
	// 				continue;
	// 			}

	// 			//fine check if particle collides with child
	// 			if (!GeoVox::geometry::GJK(_children[c_idx]->box, P)){
	// 				continue;
	// 			}

	// 			_children[c_idx]->insert_particle(P);
	// 			return;
	// 		}
	// 	}else{
	// 		if (local_particles.size() >= _root->max_particles_per_node){
	// 			// std::cout << "ID= " << ID << " local_particles.size()= " << local_particles.size() << std::endl;
	// 			new_make_children();
	// 			// std::cout << "ID= " << ID << " local_particles.size()= " << local_particles.size() << std::endl;
	// 			for (int c_idx=0; c_idx<8; c_idx++){
	// 				// std::cout << "child " << c_idx << std::endl;
	// 				_children[c_idx]->insert_particle(P);
	// 			}
	// 			return;
	// 		}else{
	// 			// std::cout << "inserted particle\n";
	// 			local_particles.push_back(P);
	// 			std::cout << ID << ": " << local_particles.size() << std::endl;
	// 			int temp_nvert = 0;
	// 			for (int v_idx=0; v_idx<8; v_idx++){
	// 				if (P.contains(box[v_idx])){
	// 					temp_nvert+=1;
	// 				}
	// 			}
	// 			_nvert = std::max(_nvert, temp_nvert);
	// 			return;
	// 		}
	// 	}
	// }


	// void AssemblyNode::new_make_children(){
	// 	// std::cout << "new_make_children\n";
	// 	if (_depth >= _root->_maxdepth){
	// 		return;
	// 	}

	// 	OctreeNode::make_children();
	// 	// std::cout << "AssemblyNode::make_children: _isdivided= " << _isdivided << std::endl;
	// 	for (long unsigned int p_idx=0; p_idx<local_particles.size(); p_idx++){
	// 		insert_particle(local_particles[p_idx]);
	// 	}

	// 	local_particles.clear();
	// }


	bool AssemblyNode::in_particle(const Point3& point) const{
		// if (_isdivided){
		// 	for(int c_idx=0; c_idx<8; c_idx++){
		// 		if (_children[c_idx]->box.contains(point)){
		// 			return _children[c_idx]->in_particle(point);
		// 		}
		// 	}
		// }

		// for (long unsigned int p_idx=0; p_idx<_particle_index.size(); p_idx++){
		// 	long unsigned int _particle_number = _particle_index[p_idx];
		// 	const SuperEllipsoid &P = _root->_particles[_particle_number];
		// 	if (P.contains(point)){
		// 		return true;
		// 	}
		// }
		
		// return false;
		// std::cout << "in_particle\n";
		AssemblyNode const* leaf = findleaf_const(point);
		for (long unsigned int p_idx=0; p_idx<leaf->_data.size(); p_idx++){
			if (_data[p_idx].contains(point)){
				return true;
			}
		}
		return false;
	}


	// void AssemblyNode::make_children(){
	// 	OctreeNode::make_children();

	// 	//add particles to children
	// 	for (int c_idx=0; c_idx<8; c_idx++){

	// 		for (long unsigned int p_idx=0; p_idx<_particle_index.size(); p_idx++){
				
	// 			long unsigned int _particle_number = _particle_index[p_idx];
	// 			const SuperEllipsoid &P = _root->_particles[_particle_number];
				
	// 			//coarse check if particle collides with child
	// 			if (!GeoVox::geometry::GJK(_children[c_idx]->box, P.bbox())){
	// 				continue;
	// 			}

	// 			//fine check if particle collides with child
	// 			if (!GeoVox::geometry::GJK(_children[c_idx]->box, P)){
	// 				continue;
	// 			}

	// 			//add index to child particle index list
	// 			_children[c_idx]->_particle_index.push_back(_particle_number);
	// 			int temp_vert = 0;
	// 			for (int v_idx=0; v_idx<8; v_idx++){
	// 				// _children[c_idx]->box[v_idx].print(std::cout);
	// 				// std::cout << "\n contained= " << P.contains(_children[c_idx]->box[v_idx]) << std::endl;

	// 				if (P.contains(_children[c_idx]->box[v_idx])){
	// 					temp_vert += 1;
	// 				}
	// 			}
	// 			_children[c_idx]->_nvert = std::max(_children[c_idx]->_nvert, temp_vert);
	// 			// _children[c_idx]->box.print(std::cout);
	// 			// std::cout << std::endl;

	// 			// std::cout << "element marker _nvert= " << _nvert << std::endl;
	// 		}
	// 	}
	// }

	// void AssemblyNode::divide(){
	// 	//TRAVERSE TO LEAF
	// 	if (_isdivided){
	// 		for (int c_idx=0; c_idx<8; c_idx++){
	// 			std::cout << "child " << c_idx << std::endl;
	// 			_children[c_idx]->divide();
	// 		}
	// 		return;
	// 	}

	// 	//ONLY DIVIDE IF THERE ARE PARTICLES
	// 	if (_particle_index.size() == 0){
	// 		return;
	// 	}

	// 	//PARTICLES ARE CONVEX, ALL CHILDREN WOULD HAVE _nvert=8
	// 	if (_nvert == 8){
	// 		return;
	// 	}

	// 	//RESPECT _maxdepth
	// 	if (_depth >= _root->_maxdepth){
	// 		return;
	// 	}

	// 	//DIVIDE
	// 	AssemblyNode::make_children();

	// 	//CONTINUE TO CHILDERN
	// 	if (_depth+1 < _root->_maxdepth){
	// 		for (int c_idx=0; c_idx<8; c_idx++){
	// 			_children[c_idx]->divide();
	// 		}
	// 	}
	// }

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


	void AssemblyNode::create_point_global_index_maps(std::vector<Point3>& points, std::map<long unsigned int, long unsigned int>& reduced_index) const{
		if (_isdivided){
			for (int c_idx=0; c_idx<8; c_idx++){
				// std::cout << "recurse from " << _ID << " to " << _children[i]->_ID << std::endl;
				_children[c_idx]->create_point_global_index_maps(points, reduced_index);
			}
		}
		else{
			//compute global index
			long unsigned int global_index[8];
			get_global_vertex_index(global_index);

			//populate maps
			for (int i=0; i<8; i++){
				// std::cout << point_map.count(global_index[i]) << std::endl;
				if (!reduced_index.count(global_index[i])){

					// std::cout << "Added global_index " << global_index[i] << " with reduced_index " << reduced_index.size() << std::endl;
					points.push_back(box[i]); //important that this is VTK_VOXEL ordering
					reduced_index[global_index[i]] = reduced_index.size();
				}
			}
		}
	}


	void AssemblyNode::makeElements(const std::map<long unsigned int, long unsigned int>& reduced_index, std::vector<std::vector<long unsigned int>> &elem2node, std::vector<int> &elemMarkers) const{
		if (_isdivided){
			for (int i=0; i<8; i++){
				_children[i]->makeElements(reduced_index, elem2node, elemMarkers);
			}
		}else{
			//set nodes
			// unsigned int vtkID = 11; //voxel default
			// bool used_nodes[26] {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

			// //check if this element is non-conforming (ASSUMES MESH IS GRADATED)
			// Point3 H = box.high() - box.low();
			// Point3 neighbor_center;
			// Node* neighbor;

			// neighbor_center = box.center() + Point3(-H[0], 0, 0);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// if (neighbor!=NULL){
			// 	if (neighbor->_isdivided){
			// 		used_nodes[20] = 1;
			// 		used_nodes[11] = 1;
			// 		used_nodes[15] = 1;
			// 		used_nodes[16] = 1;
			// 		used_nodes[19] = 1;
			// 		vtkID = 41; //OctCell (Convex Cell)
			// 	}
			// }

			// neighbor_center = box.center() + Point3(H[0], 0, 0);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// if (neighbor!=NULL){
			// 		if (neighbor->_isdivided){
			// 		used_nodes[21] = 1;
			// 		used_nodes[9]  = 1;
			// 		used_nodes[13] = 1;
			// 		used_nodes[17] = 1;
			// 		used_nodes[18] = 1;
			// 		vtkID = 41; //OctCell (Convex Cell)
			// 	}
			// }

			// neighbor_center = box.center() + Point3(0, -H[1], 0);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// if (neighbor!=NULL){
			// 	if (neighbor->_isdivided){
			// 		used_nodes[22] = 1;
			// 		used_nodes[8]  = 1;
			// 		used_nodes[12] = 1;
			// 		used_nodes[16] = 1;
			// 		used_nodes[17] = 1;
			// 		vtkID = 41; //OctCell (Convex Cell)
			// 	}
			// }

			// neighbor_center = box.center() + Point3(0, H[1], 0);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// if (neighbor!=NULL){
			// 	if (neighbor->_isdivided){
			// 		used_nodes[21] = 1;
			// 		used_nodes[9]  = 1;
			// 		used_nodes[13] = 1;
			// 		used_nodes[17] = 1;
			// 		used_nodes[18] = 1;
			// 		vtkID = 41; //OctCell (Convex Cell)
			// 	}
			// }

			// neighbor_center = box.center() + Point3(0, 0, -H[2]);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// if (neighbor!=NULL){
			// 	if (neighbor->_isdivided){
			// 		used_nodes[24] = 1;
			// 		used_nodes[8]  = 1;
			// 		used_nodes[9]  = 1;
			// 		used_nodes[10] = 1;
			// 		used_nodes[11] = 1;
			// 		vtkID = 41; //OctCell (Convex Cell)
			// 	}
			// }

			// neighbor_center = box.center() + Point3(0, 0, H[2]);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// neighbor = _root->findnode(_depth, neighbor_center);
			// if (neighbor!=NULL){
			// 	if (neighbor->_isdivided){
			// 		used_nodes[25] = 1;
			// 		used_nodes[12] = 1;
			// 		used_nodes[13] = 1;
			// 		used_nodes[14] = 1;
			// 		used_nodes[15] = 1;
			// 		vtkID = 41; //OctCell (Convex Cell)
			// 	}
			// }









			long unsigned int global_index[8];
			get_global_vertex_index(global_index);

			elem2node.push_back({reduced_index.at(global_index[0]),
								reduced_index.at(global_index[1]), 
								reduced_index.at(global_index[2]), 
								reduced_index.at(global_index[3]),
								reduced_index.at(global_index[4]),
								reduced_index.at(global_index[5]),
								reduced_index.at(global_index[6]),
								reduced_index.at(global_index[7])});

			int nvert = 0;
			for (int v_idx=0; v_idx<8; v_idx++){
				if (in_particle(box[v_idx])){
					nvert+=1;
				}
			}
			elemMarkers.push_back(size());
		}
	}

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



	void Assembly::make_tree(int maxdepth){
		if (_isdivided){
			for (int i=0; i<8; i++){
				delete _children[i];
			}
		}
		_isdivided = false;

		// _maxdepth = maxdepth;
		// divide();
		for (long unsigned int p_idx=0; p_idx<_particles.size(); p_idx++){
			std::cout << "particle " << p_idx << " of " << _particles.size() << " : ";
			_particles[p_idx].print(std::cout);
			std::cout << std::endl;
			insert_data(_particles[p_idx]);
		}
	}


	Mesh Assembly::make_voxel_mesh() const{
		//ASSEMBLE REDUCED GLOBAL INDICES
		std::vector<Point3> points;
		std::map<long unsigned int, long unsigned int> reduced_index;
		std::cout << "making point index\n";
		create_point_global_index_maps(points, reduced_index);

		// std::cout << "putting points into a new octree\n";
		// GeoVox::mesh::MeshNode meshnodes(points);

		// for (long unsigned int i=0; i<points.size(); i++){
		// 	std::cout << "Point " << i << " Point3(";
		// 	points[i].print(std::cout);
			
		// 	long unsigned int new_idx = meshnodes.find(points[i]);
		// 	std::cout << ") -> " << new_idx << " Point3(" ;
		// 	meshnodes[new_idx].print(std::cout);
		// 	std::cout << ") : difference: " << (points[i]-meshnodes[new_idx]).norm() << std::endl;
		// }


		//MAKE ELEMENTS
		std::vector<std::vector<long unsigned int>> elem2node;
		elem2node.reserve(_nleaves);

		std::vector<int> elemMarkers;
		elemMarkers.reserve(_nleaves);

		std::cout << "making elements\n";
		makeElements(reduced_index, elem2node, elemMarkers);

		//MAKE NODE MARKERS
		std::vector<int> nodeMarkers;
		nodeMarkers.reserve(points.size());
		for (long unsigned int n=0; n<points.size(); n++){
			nodeMarkers.push_back(in_particle(points[n]));
		}

		//MAKE VTK_ID
		std::vector<unsigned int> vtkElemID(_nleaves, 11);
		
		//MAKE MESH
		Mesh mesh(points, elem2node, vtkElemID, nodeMarkers, elemMarkers);



		return mesh;
	}


	void Assembly::save_geometry(const std::string filename, const Box& box, const int N[3]) const{
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

		for (int k=0; k<N[2]; k++){
			centroid[2] = box.low()[2] + H[2]*(0.5+k);
			for (int j=0; j<N[1]; j++){
				centroid[1] = box.low()[1] + H[1]*(0.5+j);
				for (int i=0; i<N[0]; i++){
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

	void Assembly::save_geometry(const std::string filename, const int N[3]) const{
		save_geometry(filename, box, N);
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