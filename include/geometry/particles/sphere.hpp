#pragma once

#include "geometry/particles/particle_base.hpp"
#include "util/point.hpp"

namespace GV
{
	struct Sphere : public Particle<Sphere>
	{
		using BASE = Particle<Sphere>;
		using Point_t = typename BASE::Point_t;

		//record radius and center
		double _r_{1};
		Point_t _center_{0,0,0};

		//querries
		inline Point_t center_impl() const {return _center_;}
		
		//
		inline Point_t to_local_impl(const Point_t& g_pt) const {return g_pt-_center_;}
		inline Point_t to_global_impl(const Point_t& l_pt) const {return l_pt+_center_;}
		inline Point_t rotate_to_local_impl(const Point_t& g_dir) const {return g_dir;}
		inline Point_t rotate_to_global_impl(const Point_t& l_dir) const {return l_dir;}
		
		bool is_in_bbox_impl(const Point_t& g_pt) const {
			const auto l_pt = to_local_impl(g_pt);
			if (std::fabs(l_pt[0]) > _r_) {return false;}
			if (std::fabs(l_pt[1]) > _r_) {return false;}
			if (std::fabs(l_pt[2]) > _r_) {return false;}
			return true;
		}
		inline bool contains_impl(const Point_t& g_pt) const {
			return g_pt[0]*g_pt[0]+g_pt[1]*g_pt[1]+g_pt[2]*g_pt[2] <= _r_*_r_; 
		}


	};
}