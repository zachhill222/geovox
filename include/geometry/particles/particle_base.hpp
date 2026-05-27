#pragma once

#include "util/point.hpp"

namespace GV
{
	template<typename DERIVED=void>
	struct Particle
	{
		using Point_t = Point<3,double>;

		//basic querry methods
		inline Point_t center() const {return static_cast<DERIVED* const>(this) -> center_impl();}
		
		//transform rotation+translation between local/global coordinates
		inline Point_t to_local(const Point_t& g_pt) const {return static_cast<DERIVED const*>(this) -> to_local_impl(g_pt);}
		inline Point_t to_global(const Point_t& l_pt) const {return static_cast<DERIVED const*>(this) -> to_global_impl(l_pt);}

		//rotation only transformations between local/global coordinates
		inline Point_t rotate_to_local(const Point_t& g_dir) const {return static_cast<DERIVED const*>(this) -> rotate_to_local_impl(g_dir);}
		inline Point_t rotate_to_global(const Point_t& l_dir) const {return static_cast<DERIVED const*>(this) -> rotate_to_global_impl(l_dir);}

		//basic region querries
		inline bool is_in_bbox(const Point_t& g_pt) const {return static_cast<DERIVED const*>(this) -> is_in_bbox_impl(g_pt);}
		inline bool contains(const Point_t& g_pt) const {return static_cast<DERIVED const*>(this) -> contains_impl(g_pt);}

		//geometry querries
		inline Point_t support(const Point_t& g_dir) const {return static_cast<DERIVED const*>(this) -> support_impl(g_dir);}
		
		//signed distance methods
		inline double signed_distance(const Point_t& g_pt) const {return static_cast<DERIVED const*>(this) -> signed_distance_impl(g_pt);}
	};
}