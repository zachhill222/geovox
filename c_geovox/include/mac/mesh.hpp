

#include "mesh/vtk_structured.hpp"

#include <vector>

using StructuredPoints = GeoVox::mesh::StructuredPoints;


namespace GeoVox::MAC{
	class MacMesh{

	public:
		//DOF masks
		StructuredPoints centroid_mask; //mask of cells to use (global indexing): Nx by Ny by Nz
		StructuredPoints u_mask; //x-velocity mask (global indexing): (Nx+1) by Ny by Nz
		StructuredPoints v_mask; //y-velocity mask (global indexing): Nx by (Ny+1) by Nz
		StructuredPoints w_mask; //z-velocity mask (global indexing): Nx by Ny by (Nz+1)

		//DOF values
		std::vector<double> p; //pressure values (reduced indexing): number of nonzeros in p_mask
		std::vector<double> u; //x-velocity values (reduced indexing): number of nonzeros in
		std::vector<double> v; //y-velocity values (reduced indexing)
		std::vector<double> w; //z-velocity values (reduced indexing)

		//index map
		
		
	};
}