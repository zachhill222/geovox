#include "mesh/mesh.hpp"

namespace GeoVox::mesh{
	long unsigned int Mesh::nTrackedNodes() const{
		long unsigned int total = 0;
		for (long unsigned int e_idx=0; e_idx<nElems(); e_idx++){
			total += _elem2node[e_idx].size();
		}
		return total;
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
			buffer << _vtkID[e_idx] << std::endl;
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

	Mesh Mesh::subdomain(const std::set<int>& eMarker) const{
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
				newMesh._vtkID.push_back(_vtkID[e_idx]);


				//add element marker
				if (_elemMarkers.size() == nElems()){
					newMesh._elemMarkers.push_back(_elemMarkers[e_idx]);
				}
			}
		}
		
		newMesh._node.shrink_to_fit();
		newMesh._elem2node.shrink_to_fit();
		newMesh._vtkID.shrink_to_fit();
		newMesh._nodeMarkers.shrink_to_fit();
		newMesh._elemMarkers.shrink_to_fit();

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