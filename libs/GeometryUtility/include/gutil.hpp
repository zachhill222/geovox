#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "geometry/polytope.hpp"
#include "geometry/basic_shapes.hpp"

#include "octree/octree_util.hpp"
#include "octree/thread_queue.hpp"
#include "octree/octree_parallel.hpp"

#include "math/fixed_point.hpp"
#include "math/float_manipulation.hpp"
#include "math/matrix.hpp"
#include "math/quaternion.hpp"
#include "math/linear_set.hpp"

#include "algorithms/convex_collision.hpp"

#include "default_octrees/point_octree.hpp"
#include "default_octrees/regular_polygon_octree.hpp"

namespace gutil
{
	template<int offset=0>
	using FixedPoint64 = FixedPoint<int64_t,offset>;
}