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
		/// Helper method to create the periodic images of an object
		/////////////////////////////////////////////////
		template<typename T>
		std::vector<T> period_images(const T& val) requires (DIMENSION==3) {
			constexpr int s0_lo = IS_PERIODIC(0) ? -1 : 0;
			constexpr int s0_hi = IS_PERIODIC(0) ?  1 : 0;
			constexpr int s1_lo = IS_PERIODIC(1) ? -1 : 0;
			constexpr int s1_hi = IS_PERIODIC(1) ?  1 : 0;
			constexpr int s2_lo = IS_PERIODIC(2) ? -1 : 0;
			constexpr int s2_hi = IS_PERIODIC(2) ?  1 : 0;

			std::vector<T> shifted_data;
			shifted_data.reserve(N_EXTENSIONS);
			point_type shift;
			for (int s0=s0_lo; s0<=s0_hi; ++s0) {
				shift[0] = static_cast<scalar_type>(s0)*period_[0];
				for (int s1=s1_lo; s1<=s1_hi; ++s1) {
					shift[1] = static_cast<scalar_type>(s1)*period_[1];
					for (int s2=s2_lo; s2<=s2_hi; ++s2) {
						if (s0==0 && s1==0 && s2==0) {continue;}	//already checked
						shift[2] = static_cast<scalar_type>(s2)*period_[2];
						
						T cpy = val + shift;
						if (intersects(root_->bbox, cpy)) { shifted_data.push_back(std::move(cpy)); }
					}
				}
			}
			return shifted_data;
		}


		/////////////////////////////////////////////////
		/// Rather than querying up to 3^N_PERIODIC_AXES points
		/// we insert additional periodic images of objects that intersect
		/// the periodic bounding box.
		/////////////////////////////////////////////////
		void push_back(value_type value) noexcept {
			push_back_range(std::span<value_type>{&value, 1});
		}

		void push_back_range(std::vector<value_type>&& values) noexcept {
			push_back_range(std::span<value_type>(values.begin(), values.end()));
		}

		template<typename T>
		[[nodiscard]] bool collides(const T& val) noexcept {
			if (BASE::collides(val)) {return true;}
			std::vector<T> images = period_images(val);
			for (const T& val : images) {
				if (BASE::collides(val)) {return true;}
			}
			return false;
		}

		void push_back_range(std::span<value_type> values) noexcept {
			//insert data and track the inserted values
			const size_t idx_start = data_.size();
			BASE::push_back_range(values);
			const size_t idx_end = data_.size();
			
			//insert periodically shifted data that intersect the larger periodic box
			std::vector<value_type> shifted_data;
			for (size_t idx=idx_start; idx<idx_end; ++idx) {
				std::vector<value_type> shifted_values = period_images(data_[idx]);
				shifted_data.insert(shifted_data.end(), shifted_values.begin(), shifted_values.end());
			}
			BASE::push_back_range(std::move(shifted_data));
		}
	};




}

