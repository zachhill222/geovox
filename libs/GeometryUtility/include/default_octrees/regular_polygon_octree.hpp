#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "geometry/basic_shapes.hpp"
#include "octree/octree_parallel.hpp"

namespace gutil {
	///Note that the data should be allowed to be put into multiple leaf nodes
	template<int N_SIDES, typename T=double, int N_DATA=32>
	class RegularPolygonOctree : public BasicParallelOctree<RegularPolygon<N_SIDES,T>, false, 2, N_DATA, T>
	{
	public:
		using BaseClass = BasicParallelOctree<RegularPolygon<N_SIDES,T>, false, 2, N_DATA, T>;
		using typename BaseClass::Data_t;
		using typename BaseClass::Box_t;

		//constructors
		constexpr RegularPolygonOctree() noexcept : BaseClass() {}
		constexpr RegularPolygonOctree(const Box_t &bbox) noexcept : BaseClass(bbox) {}

	private:
		constexpr bool isValid(const Box_t& box, const Data_t& data) const override {
			return data.intersects(box);}
		constexpr T dist2data(const Point<2,T>& point, const Data_t& data) const override {
			//distance to closest vertex as a proxy to the closest point
			T dist = normSquared(point-data[0]);
			for (int i=1; i<Data_t::n_vertices; ++i) {
				dist = std::min(dist, normSquared(point-data[i]));
			}
			return dist;
		}
	};
}