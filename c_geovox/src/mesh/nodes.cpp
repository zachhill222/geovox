#include "mesh/nodes.hpp"

namespace GeoVox::mesh{
	bool MeshNodeNode::data_valid(const Point3& point) const{
		return box.contains(point);
	}
}

