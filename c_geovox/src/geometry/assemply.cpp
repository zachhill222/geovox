#include "geometry/assembly.hpp"


namespace GeoVox::geometry{
	//NODE (OF OCTREE)
	void Node::insert_particle(const SuperEllipsoid& P){
		if (_isdivided){
			for (int i=0; i<8; i++){
				_children[i]->insert_particle(P);
			}
		}
		else{
			//check if this node collides with the bounding box (rotated prism) of the particle
			if (GeoVox::geometry::GJK(_box, P.bbox())){
				//check if this node collides with the particle itself
				if (GeoVox::geometry::GJK(_box, P)){
					if (_depth < _maxdepth){
						//create children
						_isdivided = true;
						for (int i=0; i<8; i++){
							//initialize child node
							_children[i] = new Node(_box[i],_box.center(), _depth+1); //PROBABLY NEED new Node. Would need to add delete _children[i] to destructor.
							_children[i]->_root = _root;
							_children[i]->_parent = this;
							_children[i]->_maxdepth = _maxdepth;

							//insert particle
							_children[i]->insert_particle(P);
						}
					}
					else{
						_particles.push_back(P);
					}
				}
			}
		}
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