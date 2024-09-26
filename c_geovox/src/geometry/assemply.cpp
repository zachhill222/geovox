#include "geometry/assembly.hpp"


namespace GeoVox::geometry{
	//NODE (OF OCTREE)
	void Node::divide(){
		// std::cout << "Node(" << _ID << "): _isdivided= " << _isdivided;
	 	// std::cout << " depth= " << _depth;
	 	// std::cout << " ijk= (" << _ijk[0] << ", " << _ijk[1] << ", " << _ijk[2] << ")\n";
		
		if (_isdivided){
			// std::cout << "recurse\n";
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
				if (!GeoVox::geometry::GJK(_children[i]->_box, P.bbox())){
					continue;
				}

				//fine check if particle collides with child
				// if (!GeoVox::geometry::GJK(_children[i]->_box, P)){
				// 	continue;
				// }

				//add index to child particle index list
				_children[i]->_particle_index.push_back(_particle_number);


				//update nvert
				int temp_vert = 0;
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

	// Point3 Node::voxel_vertex(const int i) const{
	// 	double H[3] {1.0, 1.0, 1.0};
		
	// }


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

	void Node::print_nvert(std::ostream& stream) const{
		if (_isdivided) {
			for (int i=0; i<8; i++){
				_children[i]->print_nvert(stream);
			}
		}else{
			stream << this->_nvert << std::endl;
		}
	}


	// ROOT
	void Root::print_vtk(std::ostream &stream) const{
		//ASSEMBLE REDUCED GLOBAL INDICES
		std::vector<Point3> point_map;
		std::map<long unsigned int, long unsigned int> reduced_index;
		create_point_global_index_maps(point_map, reduced_index);

		std::cout << "made point map\n";

		//HEADER
		stream << "# vtk DataFile Version 2.0\n";
		stream << "Octree Structure\n";
		stream << "ASCII\n\n";

		//TOPOLOGY
		stream << "DATASET UNSTRUCTURED_GRID\n";

		//POINTS
		stream << "POINTS " << point_map.size() << " float\n";
		for (long unsigned int i=0; i<point_map.size(); i++){
			point_map[i].print(stream);
			stream << std::endl;
		}
		stream << std::endl;

		std::cout << "wrote points to file\n";

		//VOXELS
		stream << "CELLS " << _root->_nleaves << " " << 9*(_root->_nleaves) << std::endl;
		print_voxel_idx(stream, reduced_index);
		stream << std::endl;

		std::cout << "wrote voxels to file\n";

		stream << "CELL_TYPES " << _root->_nleaves << std::endl;
		for (long unsigned int i=0; i<_nleaves; i++){
			stream << "11\n";
		}
		stream << std::endl;


		//NVERT DATA
		stream << "CELL_DATA " << _root->_nleaves << std::endl;
		stream << "SCALARS nvert integer\n";
		stream << "LOOKUP_TABLE default\n";
		print_nvert(stream);
		stream << std::endl;

		std::cout << "wrote nvert to file\n";


		//CONTAINED DATA
		// stream << "POINT_DATA " << point_map.size()
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

	// void Assembly::print_tree(std::ostream &stream) const{
	// 	Node::print_tree(stream);
	// }


	void Assembly::make_tree(int maxdepth){
		_setbbox();
		if (_isdivided){
			for (int i=0; i<8; i++){
				delete _children[i];
			}
		}

		_maxdepth = maxdepth;
		// for (long unsigned int i=0; i<_particles.size(); i++){
		// 	Node::insert_particle(i, _particles[i]);
		// }
		for (unsigned int i=0; i<_maxdepth+1; i++){
			std::cout << "\n################ division " << i << " ################\n\n";
			divide();
		}
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