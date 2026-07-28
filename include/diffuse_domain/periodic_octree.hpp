#pragma once

#include "gutil.hpp"

#include <array>
#include <bit>

namespace GV {

	//////////////////////////////////////////////////////
	/// Extend the VolumeOctree class to be periodic in the
	/// specified directions. Each bit in the PeriodicAxes
	/// specifies an axis that will be treated as periodic.
	//////////////////////////////////////////////////////
	template<typename VolumeType, int PeriodicAxes=0>
	struct PeriodicVolumeOctree : public gutil::VolumeOctree<VolumeType> {
		

		//////////////////////////////////////////////////
		/// Aliases and constants
		//////////////////////////////////////////////////
		using BASE = gutil::VolumeOctree<VolumeType>;
		using value_type  = typename BASE::value_type;
		using point_type  = typename BASE::point_type;
		using box_type    = typename BASE::box_type;
		using scalar_type = typename BASE::scalar_type;
		static constexpr int DIMENSION = BASE::OPTS::DIMENSION;
		static constexpr int P_AXIS = PeriodicAxes;
		static_assert( 0<=P_AXIS && P_AXIS < (1<<DIMENSION), "PeriodicVolumeOctree - invlaid axis flag");

		using BASE::data_;
		using BASE::root_;
		using BASE::intersects;

		point_type period_;	//just the diagonal of the root bounding box masked to the periodic axes
		

		//////////////////////////////////////////////////
		/// Compute normalized extensions to check
		//////////////////////////////////////////////////
		static constexpr int N_PERIODIC_AXES = std::popcount(static_cast<unsigned>(PeriodicAxes));
		static constexpr size_t N_EXTENSIONS = []() {size_t n=1; for (int i=0; i<N_PERIODIC_AXES; ++i) {n*=3;} return n;}(); //3^N_PERIODIC_AXES
		static constexpr bool IS_PERIODIC(int ax) { return PeriodicAxes & (1<<ax); }


		/////////////////////////////////////////////////
		/// Update constructors to track the period
		/////////////////////////////////////////////////
		PeriodicVolumeOctree(const box_type& bbox) : BASE{bbox}, period_{bbox.high-bbox.low} {
			for (int ax=0; ax<DIMENSION; ++ax) {
				if (!(PeriodicAxes&(1<<ax))) {period_[ax] = scalar_type{0};}
			}

			//we track objects in a larger bounding box than requested to accurately capture the period
			//this is better for querying e.g. the signed distance function to the nearest object
			// BASE::construct_root(box_type{bbox.low-period_, bbox.high+period_});
		}


		/////////////////////////////////////////////////
		/// Rather than querying up to 3^N_PERIODIC_AXES points
		/// we insert additional periodic images of objects that intersect
		/// the periodic bounding box.
		void push_back(value_type value) noexcept {
			push_back_range(std::span<value_type>{&value, 1});
		};

		void push_back_range(std::vector<value_type>&& values) noexcept {
			push_back_range(std::span<value_type>(values.begin(), values.end()));
		}

		void push_back_range(std::span<value_type> values) noexcept requires(DIMENSION==3) {
			std::cout << "Periodic" << std::endl;
			//insert data and track the inserted values
			const size_t idx_start = data_.size();
			BASE::push_back_range(values);
			const size_t idx_end = data_.size();
			
			//insert periodically shifted data that intersect the larger periodic box
			constexpr int s0_lo = IS_PERIODIC(0) ? -1 : 0;
			constexpr int s0_hi = IS_PERIODIC(0) ?  1 : 0;
			constexpr int s1_lo = IS_PERIODIC(1) ? -1 : 0;
			constexpr int s1_hi = IS_PERIODIC(1) ?  1 : 0;
			constexpr int s2_lo = IS_PERIODIC(2) ? -1 : 0;
			constexpr int s2_hi = IS_PERIODIC(2) ?  1 : 0;

			std::vector<value_type> shifted_data;
			shifted_data.reserve(idx_end-idx_start);
			point_type shift;
			for (int s0=s0_lo; s0<=s0_hi; ++s0) {
				shift[0] = static_cast<scalar_type>(s0)*period_[0];
				for (int s1=s1_lo; s1<=s1_hi; ++s1) {
					shift[1] = static_cast<scalar_type>(s1)*period_[1];
					for (int s2=s2_lo; s2<=s2_hi; ++s2) {
						if (s0==0 && s1==0 && s2==0) {continue;}	//already checked
						shift[2] = static_cast<scalar_type>(s2)*period_[2];
						
						shifted_data.clear();
						for (size_t idx=idx_start; idx<idx_end; ++idx) {
							value_type cpy = data_[idx];
							cpy.translate_by(shift);
							if (intersects(root_->bbox, cpy )) { shifted_data.push_back(std::move(cpy)); }
						}
						BASE::push_back_range(std::move(shifted_data));
					}
				}
			}
		}


		/////////////////////////////////////////////////
		/// Overwrite find_nearest and signed distance. Everything else should fall into place.
		/////////////////////////////////////////////////
		// void project_to_domain(point_type& point) const noexcept {
		// 	for (int ax=0; ax<DIMENSION; ++ax) {
		// 		if ( PeriodicAxes & (1<<ax) ) {
		// 			point[ax] = gutil::clamp_periodic(point[ax], this->root_->bbox.low[ax], this->root_->bbox.high[ax]);
		// 			assert(this->root_->bbox.low[ax]<= point[ax] && point[ax]<=this->root_->bbox.high[ax]);
		// 		}
		// 	}
		// }

		// [[nodiscard]] scalar_type signed_distance(const point_type& point) const noexcept {
		// 	point_type best;
		// 	const size_t idx = this->find_nearest(point, best);
		// 	return this->data_[idx].signed_distance(best);
		// }


		// [[nodiscard]] size_t find_nearest(point_type point, point_type& best_point) const requires(DIMENSION==3) {
		// 	//check the projected/center point
		// 	project_to_domain(point);
			
		// 	best_point = point;			
		// 	size_t best_idx = BASE::find_nearest(best_point);
		// 	scalar_type best_d2 = this->distance_sq(this->data_[best_idx], best_point);

		// 	//check the lattice of periodic extensions
		// 	constexpr int s0_lo = IS_PERIODIC(0) ? -1 : 0;
		// 	constexpr int s0_hi = IS_PERIODIC(0) ?  1 : 0;
		// 	constexpr int s1_lo = IS_PERIODIC(1) ? -1 : 0;
		// 	constexpr int s1_hi = IS_PERIODIC(1) ?  1 : 0;
		// 	constexpr int s2_lo = IS_PERIODIC(2) ? -1 : 0;
		// 	constexpr int s2_hi = IS_PERIODIC(2) ?  1 : 0;


		// 	point_type extended = point;
		// 	for (int s0=s0_lo; s0<=s0_hi; ++s0) {
		// 		extended[0] = point[0] + static_cast<scalar_type>(s0)*period_[0];
		// 		for (int s1=s1_lo; s1<=s1_hi; ++s1) {
		// 			extended[1] = point[1] + static_cast<scalar_type>(s1)*period_[1];
		// 			for (int s2=s2_lo; s2<=s2_hi; ++s2) {
		// 				if (s0==0 && s1==0 && s2==0) {continue;}	//already checked
		// 				extended[2] = point[2] + static_cast<scalar_type>(s2)*period_[2];
						
		// 				const size_t idx = BASE::find_nearest(extended);
		// 				const scalar_type d2 = this->distance_sq(this->data_[idx], extended);
		// 				if (d2<best_d2) { best_d2=d2; best_idx=idx; best_point=extended;}
		// 			}
		// 		}
		// 	}

		// 	return best_idx;
		// }
	};




}

