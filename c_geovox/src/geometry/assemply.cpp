#include "geometry/assembly.hpp"


namespace GeoVox::geometry{
	//NODE (OF OCTREE)
	bool Node::in_particle(const Point3& point) const{
		if (_isdivided){
			for(int c_idx=0; c_idx<8; c_idx++){
				if (_children[c_idx]->_box.contains(point)){
					return _children[c_idx]->in_particle(point);
				}
			}
		}

		for (long unsigned int p_idx=0; p_idx<_particle_index.size(); p_idx++){
			long unsigned int _particle_number = _particle_index[p_idx];
			const SuperEllipsoid &P = _root->_particles[_particle_number];
			if (P.contains(point)){
				return true;
			}
		}
		
		return false;
	}


	void Node::divide(){
		if (_isdivided){
			std::cout << "recurse\n";
			for (int i=0; i<8; i++){
				_children[i]->divide();
			}
			return;
		}

		if (_particle_index.size() == 0){
			// std::cout << "no particles\n";
			return;
		}

		if (_nvert == 8){
			// std::cout << "voxel contained in a particle\n";
			return;
		}

		if (_depth >= _root->_maxdepth){
			// std::cout << "maxdepth\n";
			return;
		}

		
		//create children
		_isdivided = true;
		_root->_nleaves += 7;
		// std::cout << "divide\n";

		for (int i=0; i<8; i++){
			_children[i] = new Node(_box[i], _box.center());
			_children[i]->_root = _root;
			_children[i]->_parent = this;
			_children[i]->_ID = this->_ID*8+i+1;
			_children[i]->_depth = _depth+1;
			// std::cout << "created child with ID " << _children[i]->_ID << std::endl;
		}

		//update _ijk indices for children
		_children[0]->_ijk[0] = (2*_ijk[0]);
		_children[0]->_ijk[1] = (2*_ijk[1]);
		_children[0]->_ijk[2] = (2*_ijk[2]);

		_children[1]->_ijk[0] = (2*_ijk[0])+1;
		_children[1]->_ijk[1] = (2*_ijk[1]);
		_children[1]->_ijk[2] = (2*_ijk[2]);

		_children[2]->_ijk[0] = (2*_ijk[0]);
		_children[2]->_ijk[1] = (2*_ijk[1])+1;
		_children[2]->_ijk[2] = (2*_ijk[2]);

		_children[3]->_ijk[0] = (2*_ijk[0])+1;
		_children[3]->_ijk[1] = (2*_ijk[1])+1;
		_children[3]->_ijk[2] = (2*_ijk[2]);

		_children[4]->_ijk[0] = (2*_ijk[0]);
		_children[4]->_ijk[1] = (2*_ijk[1]);
		_children[4]->_ijk[2] = (2*_ijk[2])+1;

		_children[5]->_ijk[0] = (2*_ijk[0])+1;
		_children[5]->_ijk[1] = (2*_ijk[1]);
		_children[5]->_ijk[2] = (2*_ijk[2])+1;

		_children[6]->_ijk[0] = (2*_ijk[0]);
		_children[6]->_ijk[1] = (2*_ijk[1])+1;
		_children[6]->_ijk[2] = (2*_ijk[2])+1;

		_children[7]->_ijk[0] = (2*_ijk[0])+1;
		_children[7]->_ijk[1] = (2*_ijk[1])+1;
		_children[7]->_ijk[2] = (2*_ijk[2])+1;


		//add particles to children
		for (int i=0; i<8; i++){
			for (long unsigned int j=0; j<_particle_index.size(); j++){
				long unsigned int _particle_number = _particle_index[j];
				const SuperEllipsoid &P = _root->_particles[_particle_number];
				
				//coarse check if particle collides with child
				// std::cout << "COARSE CHECK\n";
				if (!GeoVox::geometry::GJK(_children[i]->_box, P.bbox())){
					continue;
				}

				//fine check if particle collides with child
				// std::cout << "FINE CHECK\n";
				if (!GeoVox::geometry::GJK(P, _children[i]->_box)){
					continue;
				}

				//add index to child particle index list
				_children[i]->_particle_index.push_back(_particle_number);


				//update nvert
				long unsigned int temp_vert = 0;
				for (int v=0; v<8; v++){
					if (P.contains(_children[i]->_box[v])){
						temp_vert += 1;
					}
				}

				if (temp_vert > _children[i]->_nvert){
					_children[i]->_nvert = temp_vert;
				}
			}
		}

		//divide again if needed
		if (_depth+1 < _root->_maxdepth){
			for (int c_idx=0; c_idx<8; c_idx++){
				_children[c_idx]->divide();
			}
		}
	}

	void Node::print_tree(std::ostream& stream) const{
		//indent
		for (unsigned int i=0; i<_depth; i++){
			stream << "- - - - ";
		}

		stream << "Node("<<_ID << "): ijk= [" << _ijk[0] << ", " << _ijk[1] << ", " << _ijk[2] << "]\n";

		if (_isdivided){
			for (int i=0; i<8; i++){
				_children[i]->print_tree(stream);
			}
		}
	}


	

	void Node::get_global_vertex_index(long unsigned int (&global_index)[8]) const{
		//compute global index
		long unsigned int M2 = pow(2,_root->_maxdepth) + 1; //maximum number of vertices in side length
		long unsigned int Mm2 = pow(2, _root->_maxdepth-_depth); //conversion factor from current depth to _maxdepth

		// std::cout << "_depth=" << _depth << " _maxdepth=" << _root->_maxdepth << " M2=" << M2 << " Mm2=" << Mm2 << std::endl;

		global_index[0] = Mm2*(_ijk[0]   + M2*(_ijk[1]   + M2*(_ijk[2]  )));
		global_index[1] = Mm2*(_ijk[0]+1 + M2*(_ijk[1]   + M2*(_ijk[2]  )));
		global_index[2] = Mm2*(_ijk[0]   + M2*(_ijk[1]+1 + M2*(_ijk[2]  )));
		global_index[3] = Mm2*(_ijk[0]+1 + M2*(_ijk[1]+1 + M2*(_ijk[2]  )));
		global_index[4] = Mm2*(_ijk[0]   + M2*(_ijk[1]   + M2*(_ijk[2]+1)));
		global_index[5] = Mm2*(_ijk[0]+1 + M2*(_ijk[1]   + M2*(_ijk[2]+1)));
		global_index[6] = Mm2*(_ijk[0]   + M2*(_ijk[1]+1 + M2*(_ijk[2]+1)));
		global_index[7] = Mm2*(_ijk[0]+1 + M2*(_ijk[1]+1 + M2*(_ijk[2]+1)));
	}

	void Node::create_point_global_index_maps(std::vector<Point3>& point_map, std::map<long unsigned int, long unsigned int>& reduced_index) const{
		if (_isdivided){
			for (int i=0; i<8; i++){
				// std::cout << "recurse from " << _ID << " to " << _children[i]->_ID << std::endl;
				_children[i]->create_point_global_index_maps(point_map, reduced_index);
			}
		}
		else{
			//compute global index
			long unsigned int global_index[8];
			get_global_vertex_index(global_index);
			// for (int i=0; i<8; i++){
			// 	std::cout << global_index[i] << " " << std::endl;
			// }

			//populate maps
			for (int i=0; i<8; i++){
				// std::cout << point_map.count(global_index[i]) << std::endl;
				if (!reduced_index.count(global_index[i])){

					// std::cout << "Added global_index " << global_index[i] << " with reduced_index " << reduced_index.size() << std::endl;
					point_map.push_back(_box[i]); //important that this is VTK_VOXEL ordering
					reduced_index[global_index[i]] = reduced_index.size();
				}
			}
		}
	}


	void Node::print_voxel_idx(std::ostream& stream, const std::map<long unsigned int, long unsigned int>& reduced_index) const{
		if (_isdivided) {
			for (int i=0; i<8; i++){
				_children[i]->print_voxel_idx(stream, reduced_index);
			}
		}else{
			//compute global index
			long unsigned int global_index[8];
			get_global_vertex_index(global_index);

			stream << "8 "; //number of vertices
			for (int i=0; i<8; i++){
				stream << reduced_index.at(global_index[i]) << " ";
			}
			stream << std::endl;
		}
	}

	void Node::get_nvert(){
		if (_isdivided){
			_nvert = 0;
			for (int c_idx=0; c_idx<8; c_idx++){
				_children[c_idx]->get_nvert();
				_nvert += _children[c_idx]->_nvert;
			}
		}
	}

	void Node::move_to_particle_surface(Point3& point) const{
		//traverse to leaf that contains point (if it is unique)
		// or traverse to branch point
		// std::cout << _particle_index.size() << std::endl;
		if (_particle_index.size() == 0){
			return;
		}

		if (_particle_index.size() <= 5){
			// std::cout << "search\n";
			double min_level_val =  _root->_particles[_particle_index[0]].levelval(point);
			long unsigned int min_p_idx = 0;

			for (long unsigned int p_idx=1; p_idx < _particle_index.size(); p_idx++){
				SuperEllipsoid &P = _root->_particles[_particle_index[p_idx]];
				if (P.levelval(point) < min_level_val){
					min_p_idx = p_idx;
				}
			}

			SuperEllipsoid &P = _root->_particles[_particle_index[min_p_idx]];
			point = P.closest_point(point);
			return;
		}

		// std::cout << "recurse\n";
		for (int c_idx=0; c_idx<8; c_idx++){
			if (_children[c_idx]->_box.contains(point)){
				_children[c_idx]->move_to_particle_surface(point);
				return;
			}
		}
	}

	void Node::print_nvert(std::ostream& stream) const{
		if (_isdivided) {
			for (int i=0; i<8; i++){
				_children[i]->print_nvert(stream);
			}
		}else{
			stream << this->_nvert << std::endl;
		}
	}

	void Node::makeElements(const std::map<long unsigned int, long unsigned int>& reduced_index, std::vector<std::vector<long unsigned int>> &elem2node, std::vector<int> &elemMarkers) const{
		if (_isdivided){
			for (int i=0; i<8; i++){
				_children[i]->makeElements(reduced_index, elem2node, elemMarkers);
			}
		}else{
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

			elemMarkers.push_back(_nvert);
		}
	}


	// ROOT
	void Root::print_vtk(std::ostream &stream) const{
		//ASSEMBLE REDUCED GLOBAL INDICES
		std::vector<Point3> point_map;
		std::map<long unsigned int, long unsigned int> reduced_index;
		create_point_global_index_maps(point_map, reduced_index);


		//create string stream for fewer io writes
		std::stringstream buffer;

		std::cout << "made point map\n";

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << "Octree Structure\n";
		buffer << "ASCII\n\n";

		//TOPOLOGY
		buffer << "DATASET UNSTRUCTURED_GRID\n";

		//POINTS
		buffer << "POINTS " << point_map.size() << " float\n";
		for (long unsigned int i=0; i<point_map.size(); i++){
			point_map[i].print(buffer);
			buffer << std::endl;
		}
		buffer << std::endl;

		stream << buffer.rdbuf();
		buffer.str("");

		std::cout << "wrote points to file\n";

		//VOXELS
		buffer << "CELLS " << _root->_nleaves << " " << 9*(_root->_nleaves) << std::endl;
		print_voxel_idx(buffer, reduced_index);
		buffer << std::endl;

		stream << buffer.rdbuf();
		buffer.str("");

		std::cout << "wrote voxels to file\n";

		buffer << "CELL_TYPES " << _root->_nleaves << std::endl;
		for (long unsigned int i=0; i<_nleaves; i++){
			buffer << "11\n";
		}
		buffer << std::endl;


		//NVERT DATA
		buffer << "CELL_DATA " << _root->_nleaves << std::endl;
		buffer << "SCALARS nvert integer\n";
		buffer << "LOOKUP_TABLE default\n";
		print_nvert(buffer);
		buffer << std::endl;

		stream << buffer.rdbuf();
		buffer.str("");

		std::cout << "wrote nvert to file\n";


		//CONTAINED DATA
		buffer << "POINT_DATA " << point_map.size() << std::endl;
		buffer << "SCALARS in_particle integer\n";
		buffer << "LOOKUP_TABLE default\n";
		for (long unsigned int i=0; i<point_map.size(); i++){
			buffer << in_particle(point_map[i]) << std::endl;
		}
		buffer << std::endl;

		stream << buffer.rdbuf();
		buffer.str("");

	}

	void Root::save_vtk(const std::string fullfile) const{
		std::ofstream voxel_mesh(fullfile);

		if( not voxel_mesh.is_open() )
		{
			std::cout << "Couldn't write to " << fullfile << std::endl;
			voxel_mesh.close();
			return;
		}

		print_vtk(voxel_mesh);
		

		// close file
		voxel_mesh.close();
	}


	//ASSEMBLY
	SuperEllipsoid Assembly::operator[](int idx) const{
		return _particles[idx];
	}

	SuperEllipsoid& Assembly::operator[](int idx){
		return _particles[idx];
	}

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
				_particle_index.push_back(particle_number);
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

		_maxdepth = maxdepth;
		divide();
	}


	Mesh Assembly::make_voxel_mesh() const{
		//ASSEMBLE REDUCED GLOBAL INDICES
		std::vector<Point3> points;
		std::map<long unsigned int, long unsigned int> reduced_index;
		create_point_global_index_maps(points, reduced_index);

		//MAKE ELEMENTS
		std::vector<std::vector<long unsigned int>> elem2node;
		elem2node.reserve(_nleaves);

		std::vector<int> elemMarkers;
		elemMarkers.reserve(_nleaves);

		makeElements(reduced_index, elem2node, elemMarkers);

		//MAKE NODE MARKERS
		std::vector<int> nodeMarkers;
		nodeMarkers.reserve(points.size());
		for (long unsigned int n=0; n<points.size(); n++){
			nodeMarkers.push_back(in_particle(points[n]));
		}

		//MAKE VTK_ID
		std::vector<unsigned int> vtkID(_nleaves, 11);
		
		return Mesh(points, elem2node, vtkID, nodeMarkers, elemMarkers);
	}

	Mesh Assembly::make_mixed_mesh() const{
		//get initial mesh
		Mesh mesh = make_voxel_mesh();

		//mark points to move to boundary
		for (long unsigned int e_idx=0; e_idx<mesh.nElems(); e_idx++){
			if (mesh._elemMarkers[e_idx]>0 && mesh._elemMarkers[e_idx]<8){
				// std::cout << "changing boundary element " << e_idx << std::endl;
				
				if (mesh._elemMarkers[e_idx] <= 3){
					//most nodes are outside. Move inside nodes to boundary.
					// for (int n_idx=0; n_idx<8; n_idx++){
					// 	if (mesh._nodeMarkers[mesh._elem2node[e_idx][n_idx]] == 1){
					// 		// std::cout << "marking node " << mesh._elem2node[e_idx][n_idx] << " to move\n";
					// 		mesh._nodeMarkers[mesh._elem2node[e_idx][n_idx]] = 2;
					// 	}
					// }
				}else if (mesh._elemMarkers[e_idx] >=6){
					//most nodes are inside. Move outside nodes to boundary.
					for (int n_idx=0; n_idx<8; n_idx++){
						if (mesh._nodeMarkers[mesh._elem2node[e_idx][n_idx]] == 0){
							// std::cout << "marking node " << mesh._elem2node[e_idx][n_idx] << " to move\n";
							mesh._nodeMarkers[mesh._elem2node[e_idx][n_idx]] = 2;
						}
					}
				}

				//change vtkID from voxel to hexahedron
				// std::cout << "re-ordering element " << e_idx << " from voxel to hexahedron\n";
				mesh._vtkID[e_idx] = 12;

				//re-order nodes
				long unsigned int temp;

				temp = mesh._elem2node[e_idx][3];
				mesh._elem2node[e_idx][3] = mesh._elem2node[e_idx][2];
				mesh._elem2node[e_idx][2] = temp;

				temp = mesh._elem2node[e_idx][7];
				mesh._elem2node[e_idx][7] = mesh._elem2node[e_idx][6];
				mesh._elem2node[e_idx][6] = temp;
			}
		}



		//move points to boundary
		for (long unsigned int n_idx=0; n_idx<mesh.nNodes(); n_idx++){
			if (mesh._nodeMarkers[n_idx] == 2){
				// std::cout << "moving node " << n_idx << std::endl;
				move_to_particle_surface(mesh._node[n_idx]);
			}
		}

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
		save_geometry(filename, _box, N);
	}


	void Assembly::_setbbox() {
		if (_particles.size() == 0){
			_box = Box(Point3(0,0,0), Point3(1,1,1));
		}
		else{
			_box = _particles[0].axis_alligned_bbox();
			for (unsigned int i=0; i<_particles.size(); i++){
				_box.combine(_particles[i].axis_alligned_bbox());
			}
		}
	}
}