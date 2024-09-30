#include "mesh/mesh.hpp"

namespace GeoVox::mesh{

	long unsigned int Mesh::nTrackedNodes() const{
		long unsigned int total = 0;
		for (long unsigned int e_idx=0; e_idx<nElems(); e_idx++){
			total += _elem2node[e_idx].size();
		}
		return total;
	}

	void Mesh::make_node2elem(){
		_node2elem.reserve(nNodes());
		for (long unsigned int e_idx=0; e_idx<nElems(); e_idx++){
			for (long unsigned int n_idx=0; n_idx<_elem2node[e_idx].size(); n_idx++){
				_node2elem[_elem2node[e_idx][n_idx]].push_back(e_idx);
			}
		}
	}

	void Mesh::make_boundary(){
		std::vector<std::vector<long unsigned int>> temp_face2node;
		std::vector<std::vector<long unsigned int>>	temp_elem2face;
		std::vector<std::vector<long unsigned int>>	temp_face2elem;
		std::vector<unsigned int> temp_vtkFaceID;

		temp_face2node.reserve(6*nElems()); //just a guess
		temp_elem2face.resize(nElems());
		temp_face2elem.reserve(6*nElems()); //just a guess
		temp_vtkFaceID.reserve(6*nElems()); //just a guess

		//get total number of faces (double count interior)
		std::map<std::vector<long unsigned int>, long unsigned int> idx2face; //get face from SORTED vertex indices
		long unsigned int nfaces;
		for (long unsigned int e_idx=0; e_idx<nElems(); e_idx++){
			std::vector<long unsigned int> face_nodes;
			std::vector<long unsigned int> sorted_face_nodes;

			// std::cout << "checking faces on element " << e_idx << std::endl;

			switch (_vtkElemID[e_idx]){
			case 11:
				for (int i=0; i<6; i++){
					GeoVox::mesh::Voxel element(_elem2node[e_idx]);
					face_nodes = element.getface(i);

					sorted_face_nodes = face_nodes;
					std::sort(sorted_face_nodes.begin(), sorted_face_nodes.end());

					if (idx2face.count(sorted_face_nodes) == 0){
						nfaces = idx2face.size();
						idx2face[sorted_face_nodes] = nfaces;
						temp_vtkFaceID.push_back(8); //face of voxel is pixel
						temp_face2node.push_back(face_nodes);
						temp_elem2face[e_idx].push_back(nfaces);
						temp_face2elem.push_back({e_idx});
					}
					else{
						temp_elem2face[e_idx].push_back(idx2face.at(sorted_face_nodes));
						temp_face2elem[idx2face.at(sorted_face_nodes)].push_back(e_idx);
					}
				}
				break;
			case 12:
				for (int i=0; i<6; i++){
					GeoVox::mesh::Hexahedron element(_elem2node[e_idx]);
					face_nodes = element.getface(i);
					
					sorted_face_nodes = face_nodes;
					std::sort(sorted_face_nodes.begin(), sorted_face_nodes.end());
					
					//check if face center is a hanging node and skip
					bool isHanging = false;
					Point3 center = 0.25*(_node[element[0]]	+ _node[element[1]] + _node[element[2]] + _node[element[3]]);

					for (long unsigned int n_idx=0; n_idx<nNodes(); n_idx++){
						if (_node[n_idx]==center){
							isHanging = true;
							break;
						}
					}

					if (isHanging){
						continue;
					}

					if (idx2face.count(sorted_face_nodes) == 0){
						nfaces = idx2face.size();
						idx2face[sorted_face_nodes] = nfaces;
						temp_vtkFaceID.push_back(9); //face of hexahedron is quad
						temp_face2node.push_back(face_nodes);
						temp_elem2face[e_idx].push_back(nfaces);
						temp_face2elem.push_back({e_idx});
					}
					else{
						temp_elem2face[e_idx].push_back(idx2face.at(sorted_face_nodes));
						temp_face2elem[idx2face.at(sorted_face_nodes)].push_back(e_idx);
					}
				}
				break;
			default:
				std::runtime_error("Unknown _vtkElemID");
			}
		}

		temp_face2node.shrink_to_fit();
		temp_face2elem.shrink_to_fit();
		temp_vtkFaceID.shrink_to_fit();

		//determine which faces are on the boundary
		_face2node.reserve(temp_face2node.size());
		_vtkFaceID.reserve(temp_face2node.size());
		for (long unsigned int f_idx=0; f_idx<temp_face2node.size(); f_idx++){
			if (temp_face2elem[f_idx].size() == 1){
				//check if face has a hanging node
				Point3 center_point = Point3(0.0, 0.0, 0.0);
				for (long unsigned int n_idx=0; n_idx<temp_face2node[f_idx].size(); n_idx++){
					center_point += _node[temp_face2node[f_idx][n_idx]];
				}

				Point3 center = center_point/temp_face2node.size();

				//loop through _node to see if center is contained
				bool isHanging = false;
				for (long unsigned int n_idx=0; n_idx<nNodes(); n_idx++){
					if (_node[n_idx]==center){
						isHanging = true;
						std::cout << "face " << f_idx << " has a hanging center\n";
						break;
					}
				}


				if (not isHanging){
					_face2node.push_back(temp_face2node[f_idx]);
					_vtkFaceID.push_back(temp_vtkFaceID[f_idx]);
				}
			}
		}

		_face2node.shrink_to_fit();
		_vtkFaceID.shrink_to_fit();
	}

	void Mesh::saveas(const std::string filename) const{
		if (nElems()*nNodes() == 0){
			std::cout << "The mesh is empty. Did not write to " << filename << std::endl;
			return;
		}
		

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
		buffer << "DATASET UNSTRUCTURED_GRID\n";

		//POINTS
		buffer << "POINTS " << nNodes() << " float\n";
		for (long unsigned int n_idx=0; n_idx<nNodes(); n_idx++){
			_node[n_idx].print(buffer);
			buffer << std::endl;
		}
		buffer << std::endl;

		meshfile << buffer.rdbuf();
		buffer.str("");


		//ELEMENTS
		buffer << "CELLS " << nElems() << " " << nElems() + nTrackedNodes() << std::endl;
		for (long unsigned int e_idx=0; e_idx<nElems(); e_idx++){
			buffer << _elem2node[e_idx].size();
			for (long unsigned int n_idx=0; n_idx<_elem2node[e_idx].size(); n_idx++){
				buffer << " " << _elem2node[e_idx][n_idx];
			}
			buffer << std::endl;
		}
		buffer << std::endl;

		meshfile << buffer.rdbuf();
		buffer.str("");


		//VTK_IDS
		buffer << "CELL_TYPES " << nElems() << std::endl;
		for (long unsigned int e_idx=0; e_idx<nElems(); e_idx++){
			buffer << _vtkElemID[e_idx] << std::endl;
		}
		buffer << std::endl;

		meshfile << buffer.rdbuf();
		buffer.str("");


		//ELEMENT_MARKERS
		if (_elemMarkers.size() == nElems()){
			buffer << "CELL_DATA " << nElems() << std::endl;
			buffer << "SCALARS elemMarkers integer\n";
			buffer << "LOOKUP_TABLE default\n";
			for (long unsigned int e_idx=0; e_idx<nElems(); e_idx++){
				buffer << _elemMarkers[e_idx] << std::endl;
			}
			buffer << std::endl;

			meshfile << buffer.rdbuf();
			buffer.str("");
		}


		//NODE_MARKERS
		if (_nodeMarkers.size() == nNodes()){
			buffer << "POINT_DATA " << nNodes() << std::endl;
			buffer << "SCALARS nodeMarkers integer\n";
			buffer << "LOOKUP_TABLE default\n";
			for (long unsigned int n_idx=0; n_idx<nNodes(); n_idx++){
				buffer << _nodeMarkers[n_idx] << std::endl;
			}
			buffer << std::endl;

			meshfile << buffer.rdbuf();
			buffer.str("");
		}
		

		//////////////// CLOSE FILE ////////////////
		meshfile.close();
	}

	Mesh Mesh::mesh_subdomain(const std::set<int>& eMarker) const{
		Mesh newMesh = Mesh(nNodes(), nElems());
		
		if (_elemMarkers.size() == nElems()){
			newMesh._elemMarkers.reserve(nElems());
		}

		if (_nodeMarkers.size() == nNodes()){
			newMesh._nodeMarkers.reserve(nNodes());
		}

		std::map<long unsigned int, long unsigned int> old2new;
		std::vector<long unsigned int> newElem;

		for (long unsigned int e_idx=0; e_idx<nElems(); e_idx++){
			if (eMarker.count(_elemMarkers[e_idx])>0){
				//update elements
				std::vector<long unsigned int> thisElemMarker;
				newElem.resize(_elem2node[e_idx].size());


				//add nodes and index map
				for (long unsigned int n_idx=0; n_idx<_elem2node[e_idx].size(); n_idx++){
					//update nodes
					if (not old2new.count(_elem2node[e_idx][n_idx])){

						newMesh._node.push_back(_node[_elem2node[e_idx][n_idx]]); //add node to new mesh
						old2new[_elem2node[e_idx][n_idx]] = old2new.size(); //track node index change

						if (_nodeMarkers.size() == nNodes()){
							newMesh._nodeMarkers.push_back(_nodeMarkers[_elem2node[e_idx][n_idx]]);
						}
					}

					newElem[n_idx] = old2new[_elem2node[e_idx][n_idx]];
				}


				//add element to new mesh
				newMesh._elem2node.push_back(newElem);
				newMesh._vtkElemID.push_back(_vtkElemID[e_idx]);


				//add element marker
				if (_elemMarkers.size() == nElems()){
					newMesh._elemMarkers.push_back(_elemMarkers[e_idx]);
				}
			}
		}
		
		newMesh._node.shrink_to_fit();
		newMesh._elem2node.shrink_to_fit();
		newMesh._vtkElemID.shrink_to_fit();
		newMesh._nodeMarkers.shrink_to_fit();
		newMesh._elemMarkers.shrink_to_fit();

		return newMesh;
	}

	Mesh Mesh::boundary_mesh(const std::set<int>& eMarker) const{
		Mesh subdomain = mesh_subdomain(eMarker);
		return subdomain.extract_boundary_mesh();
	}

	Mesh Mesh::extract_boundary_mesh(){
		Mesh newMesh = Mesh(nNodes(), nElems());
		
		if (nFaces()==0){
			make_boundary();
		}


		//reduce index to nodes only on boundary and populate _node
		std::map<long unsigned int, long unsigned int> old2new;
		for (long unsigned int f_idx=0; f_idx<nFaces(); f_idx++){
			for (long unsigned int n_idx=0; n_idx<_face2node[f_idx].size(); n_idx++){
				if (old2new.count(_face2node[f_idx][n_idx]) == 0){
					long unsigned int old_node = _face2node[f_idx][n_idx];

					old2new[old_node] = old2new.size();
					newMesh._node.push_back(_node[old_node]);
					newMesh._nodeMarkers.push_back(_nodeMarkers[old_node]);
				}
			}
		}

		newMesh._node.shrink_to_fit();
		newMesh._nodeMarkers.shrink_to_fit();


		//populate _elem2node
		newMesh._elem2node = _face2node;
		newMesh._vtkElemID = _vtkFaceID;

		//re-label nodes
		for (long unsigned int e_idx=0; e_idx<newMesh.nElems(); e_idx++){
			for (long unsigned int n_idx=0; n_idx<newMesh._elem2node[e_idx].size(); n_idx++){
				long unsigned int old_node = newMesh._elem2node[e_idx][n_idx];
				newMesh._elem2node[e_idx][n_idx] = old2new[old_node];
			}
		}

		return newMesh;
	}

	Box Mesh::box() const{
		Point3 low;
		Point3 high;

		for (long unsigned int n_idx=0; n_idx<nNodes(); n_idx++){
			low = GeoVox::util::el_min(low, _node[n_idx]);
			high = GeoVox::util::el_max(high, _node[n_idx]);
		}

		return Box(low, high);
	}
}

