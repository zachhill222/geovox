#include "geometry/assembly.hpp"


namespace GeoVox::geometry{
	//NODE (OF OCTREE)

	void Node::insert_particle(const long unsigned int idx, const SuperEllipsoid& P){
		// std::cout << "Node(" << _ID << "):\n";

		//check if this node collides with the bounding box (rotated prism) of the particle
		if (!GeoVox::geometry::GJK(_box, P.bbox())){
			return;
		}

		// std::cout << "\tCoarse collision\n";

		//check if this node collides with the particle itself
		if (!GeoVox::geometry::GJK(_box, P)){
			return;
		}

		// std::cout << "\tFine collision\n";


		//particle collides with this node's box
		if (_isdivided){
			for (int i=0; i<8; i++){
				_children[i]->insert_particle(idx, P);
			}
		}
		else{
			//check if entire voxel is contained in P
			bool iscontained = true;
			for (int v_idx=0; v_idx<8; v_idx++){
				if (!P.contains(_box[v_idx])){
					iscontained = false;
				}
			}

			if (_depth < _maxdepth && !iscontained){
				_isdivided = true;

				//create children
				for (int i=0; i<8; i++){
					//initialize child node
					_children[i] = new Node(_box[i],_box.center(), _depth+1);
					_children[i]->_root = _root;
					_children[i]->_parent = this;
					_children[i]->_maxdepth = _maxdepth;
					_children[i]->_ID = this->_ID*8+i+1;
				}

				//update _IJK indices for children
				_children[0]->_IJK[0] = (2*this->_IJK[0]);
				_children[0]->_IJK[1] = (2*this->_IJK[1]);
				_children[0]->_IJK[2] = (2*this->_IJK[2]);

				_children[1]->_IJK[0] = (2*this->_IJK[0])+1;
				_children[1]->_IJK[1] = (2*this->_IJK[1]);
				_children[1]->_IJK[2] = (2*this->_IJK[2]);

				_children[2]->_IJK[0] = (2*this->_IJK[0]);
				_children[2]->_IJK[1] = (2*this->_IJK[1])+1;
				_children[2]->_IJK[2] = (2*this->_IJK[2]);

				_children[3]->_IJK[0] = (2*this->_IJK[0])+1;
				_children[3]->_IJK[1] = (2*this->_IJK[1])+1;
				_children[3]->_IJK[2] = (2*this->_IJK[2]);

				_children[4]->_IJK[0] = (2*this->_IJK[0]);
				_children[4]->_IJK[1] = (2*this->_IJK[1]);
				_children[4]->_IJK[2] = (2*this->_IJK[2])+1;

				_children[5]->_IJK[0] = (2*this->_IJK[0])+1;
				_children[5]->_IJK[1] = (2*this->_IJK[1]);
				_children[5]->_IJK[2] = (2*this->_IJK[2])+1;

				_children[6]->_IJK[0] = (2*this->_IJK[0]);
				_children[6]->_IJK[1] = (2*this->_IJK[1])+1;
				_children[6]->_IJK[2] = (2*this->_IJK[2])+1;

				_children[7]->_IJK[0] = (2*this->_IJK[0])+1;
				_children[7]->_IJK[1] = (2*this->_IJK[1])+1;
				_children[7]->_IJK[2] = (2*this->_IJK[2])+1;


				//update _global_vtk_voxel_idx for children
				



				//insert particle into children nodes
				for (int i=0; i<8; i++){
					_children[i]->insert_particle(idx, P);
				}

			}
			else{
				_particle_index.push_back(idx);
			}
		}
	}

	void Node::print_tree(std::ostream& stream) const{
		//indent
		for (int i=0; i<_depth; i++){
			stream << "- - - - ";
		}

		stream << "Node("<<_ID << "): IJK= [" << _IJK[0] << ", " << _IJK[1] << ", " << _IJK[2] << "]\n";

		if (_isdivided){
			for (int i=0; i<8; i++){
				_children[i]->print_tree(stream);
			}
		}
	}


	Point3 Node::global_vtk_voxel_coords(const int idx) const{
		Point3 global_box_size = _root->_box.high() - _root->_box.low();
		
		double C = pow(2,-_depth);
		double _h[3] {C*global_box_size[0], C*global_box_size[1], C*global_box_size[2]};

		Point3 result = _root->_box.low() + Point3(_h[0]*_IJK[0], _h[1]*_IJK[1], _h[2]*_IJK[2]);
		return result;
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
				SuperEllipsoid P = SuperEllipsoid(Point3(rx,ry,rz), eps1, eps2, Point3(x,y,z), Quaternion(q0,q1,q2,q3));
				_particles.push_back(P);
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

	void Assembly::print_tree(std::ostream &stream) const{
		Node::print_tree(stream);
	}


	void Assembly::make_tree(){
		_setbbox();
		for (long unsigned int i=0; i<_particles.size(); i++){
			Node::insert_particle(i, _particles[i]);
		}
	}

	void Assembly::maxdepth(const int depth){
		_maxdepth = depth;
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