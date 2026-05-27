#pragma once

#include "geometry/particles/particle_base.hpp"
#include "util/point.hpp"

#include <cmath>

namespace GV
{
	struct Sphere : public Particle<Sphere>
	{
		using BASE = Particle<Sphere>;
		using Point_t = typename BASE::Point_t;

		//record radius and center
		double _r_{1};
		Point_t _center_{0,0,0};

		//transformations
		inline Point_t to_local_impl(const Point_t& g_pt) const {return g_pt-_center_;}
		inline Point_t to_global_impl(const Point_t& l_pt) const {return l_pt+_center_;}
		inline Point_t rotate_to_local_impl(const Point_t& g_dir) const {return g_dir;}
		inline Point_t rotate_to_global_impl(const Point_t& l_dir) const {return l_dir;}
		
		//simple queries
		inline Point_t center_impl() const {return _center_;}
		
		bool is_in_bbox_impl(const Point_t& g_pt) const {
			const auto l_pt = to_local_impl(g_pt);
			if (std::fabs(l_pt[0]) > _r_) {return false;}
			if (std::fabs(l_pt[1]) > _r_) {return false;}
			if (std::fabs(l_pt[2]) > _r_) {return false;}
			return true;
		}
		
		inline bool contains_impl(const Point_t& g_pt) const {return g_pt[0]*g_pt[0]+g_pt[1]*g_pt[1]+g_pt[2]*g_pt[2] <= _r_*_r_;}
		
		//geometry queries
		Point_t support_impl(const Point_t& g_dir) const {
			//TODO: if we move to floats, this can be done with an intrinsic for 1/sqrt(x)
			const double s = _r_ / std::sqrt(g_dir[0]*g_dir[0] + g_dir[1]*g_dir[1] + g_dir[2]*g_dir[2]);
			return _center_ + s*g_dir;
		}

		double signed_distance_impl(const Point_t& g_pt) const {
			const Point_t l_pt = to_local_impl(g_pt);
			const double R     = std::sqrt(l_dir[0]*l_dir[0] + l_dir[1]*l_dir[1] + l_dir[2]*l_dir[2]);
			return R-_r_;
		}
	};
}