#pragma once

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "octree/octree_parallel.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace gutil {
	///Octree for points in space.
	template<int dim=3, typename T=double, int n_data=32, typename U=T>
	class PointOctree : public BasicParallelOctree<
									Point<dim,T>,
									true,
									dim,
									n_data,
									U>
	{
	public:
		using BaseClass = BasicParallelOctree<
									Point<dim,T>,
									true,
									dim,
									n_data,
									U>;
		using typename BaseClass::Data_t;
		using typename BaseClass::Box_t;

		//constructors
		constexpr PointOctree() noexcept : BaseClass() {}
		constexpr PointOctree(const Box_t &bbox) noexcept : BaseClass(bbox) {}

		//override find closest data
		template<typename W>
		size_t closest_point_idx(const Point<dim,W>& point) const {
			assert(this->size()>0);
			size_t idx=0;

			Point<dim,U> octree_point = static_cast<Point<dim,U>>(point);
			Point<dim,U> octree_data  = static_cast<Point<dim,U>>(this->_data[0]);
			U dist2 = squaredNorm(octree_data-octree_point);

			_recursive_find_closest_point(this->_root, octree_point, idx, dist2);
			return idx;
		}

		//find with a tolerance.
		size_t find_tol(const Point<dim,T>& point, const T tol) const {
			const Point<dim,T> H(tol);
			const Box<dim,T> box(point-H, point+H);
			std::vector<size_t> candidates = this->get_data_in_box(box);
			for (size_t idx : candidates) {
				if (squaredNorm(this->_data[idx] - point) <= tol*tol) {
					return idx;
				}
			}

			return (size_t) -1;
		}

	private:
		constexpr bool isValid(const Box_t& box, const Data_t& data) const override {return box.contains(static_cast<Point<dim,U>>(data));}
		constexpr T dist2data(const Point<dim,T>& point, const Data_t& data) const override {return squaredNorm(point-data);}
		void _recursive_find_closest_point(const typename BaseClass::Node_t* node, const Point<dim,U>& point, size_t& idx, U& dist2) const;
	};


	template<int dim, typename T, int n_data, typename U>
	void PointOctree<dim,T,n_data,U>::_recursive_find_closest_point(
		const typename PointOctree<dim,T,n_data,U>::BaseClass::Node_t* node,
		const Point<dim,U>& point,
		size_t& idx,
		U& dist2) const
	{
		assert(node);

		//engage shared lock on the way down
		std::shared_lock<std::shared_mutex> read_lock(node->mutex);

		if (!isLeaf(node))
		{
			//sort children to recurse into closest first
			std::array<std::pair<T,int>, this->N_CHILDREN> children_dist_pair;
			for (int c=0; c<this->N_CHILDREN; c++) {
				children_dist_pair[c] = {distance_squared(node->children[c]->bbox, point), c};
			}
			std::sort(children_dist_pair.begin(), children_dist_pair.end()); //sorts lexigraphically, distance is first

			//find closest child
			for (const auto& [dist2child, c] : children_dist_pair) {
				if (dist2child < dist2) {
					_recursive_find_closest_point(node->children[c], point, idx, dist2);
				}
			}
		}
		else
		{
			//check if we are the closest leaf with data
			for (int i=0; i<node->cursor; i++) {
				const Point<dim,U> data_point = static_cast<Point<dim,U>>(this->_data[node->data_idx[i]]);
				U temp_dist2 = squaredNorm(data_point - point);
				if (temp_dist2 < dist2) {
					dist2 = temp_dist2;
					idx = node->data_idx[i];
				}
			}
		}
	}
}