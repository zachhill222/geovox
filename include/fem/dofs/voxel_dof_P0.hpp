#pragma once

#include "fem/dofs/voxel_dof_base.hpp"
#include "mesh/keys/voxel_key.hpp"

#include <array>
#include <cstdint>
#include <omp.h>

namespace GV
{
	//P0 dofs (constant on each element)
	template<VoxelElementKeyType Key_type>
	struct VoxelP0 : public VoxelDOFBase<Key_type, VoxelP0<Key_type>>
	{
		//get types from the Base class
		using Base = VoxelDOFBase<Key_type, VoxelP0<Key_type>>;
		using RefPoint_t = typename Base::RefPoint_t;
		using GeoPoint_t = typename Base::GeoPoint_t;
		using Key_t      = typename Base::Key_t;
		using QuadElem_t = typename Base::QuadElem_t;

		//constants
		static constexpr uint64_t N_CHILDREN     = 8;
		static constexpr uint64_t N_DOF_PER_ELEM = 1;
		static constexpr uint64_t N_SUPPORT_ELEM = 1;
		static constexpr uint64_t N_PARENTS      = 1;

		//access the data
		using Base::key;

		//use the base constructors
		using Base::Base;

		//evaluate always to 1
		static constexpr double eval(QuadElem_t support_elem, const RefPoint_t& xi) {return 1.0;}

		//gradient is always 0
		static constexpr RefPoint_t grad(QuadElem_t support_elem, const RefPoint_t& xi) {return {0.0, 0.0, 0.0};}

		//vectorized evaluation (for a consistent interface across dof types)
		template<int N> requires (N>0)
		void eval(	std::array<double,N>&       vl, //values 
					QuadElem_t             el, //support element
					const std::array<double,N>& qx, //reference/quadrature points
					const std::array<double,N>& qy, 
					const std::array<double,N>& qz) const {
			//check that the index logic is correct
			assert(el.is_valid());
			assert(key.i() - el.i() <= 1);
			assert(key.j() - el.j() <= 1);
			assert(key.k() - el.k() <= 1);
			
			//constant evaluation
			vl.fill(1.0);
		}

		//vectorized grad (for a consistent interface across dof types)
		template<int N> requires (N>0)
		void grad(	std::array<double,N>&		gx, //gradient result
					std::array<double,N>& 		gy, 
					std::array<double,N>& 		gz, 
					QuadElem_t  	 	    	el, //support element
					const std::array<double,N>& qx, //reference/quadratrue points
					const std::array<double,N>& qy, 
					const std::array<double,N>& qz) const {

			//check that the index logic is correct
			assert(el.is_valid());
			assert(key.i() - el.i() <= 1);
			assert(key.j() - el.j() <= 1);
			assert(key.k() - el.k() <= 1);
			
			//constant evaluation
			gx.fill(0.0);
			gy.fill(0.0);
			gz.fill(0.0);
		}

		//get the support
		constexpr std::array<QuadElem_t,1> support_impl() const {
			return {key};
		}

		//refinement operations. periodic template is only for conforming to the standard dof interface
		constexpr std::array<VoxelP0,8> children_impl() const {
			const uint64_t ii=2*key.i(), jj=2*key.j(), kk=2*key.k(), dd=key.depth()+1;
			assert(dd<Key_t::MAX_DEPTH);
			return {
				VoxelP0{Key_t{dd, ii,   jj,   kk  }},
				VoxelP0{Key_t{dd, ii+1, jj,   kk  }},
				VoxelP0{Key_t{dd, ii,   jj+1, kk  }},
				VoxelP0{Key_t{dd, ii+1, jj+1, kk  }},
				VoxelP0{Key_t{dd, ii,   jj,   kk+1}},
				VoxelP0{Key_t{dd, ii+1, jj,   kk+1}},
				VoxelP0{Key_t{dd, ii,   jj+1, kk+1}},
				VoxelP0{Key_t{dd, ii+1, jj+1, kk+1}}
			};
		}

		static constexpr std::array<double,8> child_coef_impl() {
			return {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
		}

		constexpr std::array<VoxelP0,1> parents_impl() const {
			if (key.depth()==0) {return {VoxelP0{}};}
			return {VoxelP0{key.parent()}};
		}

		static constexpr std::array<double,1> parent_coefs_impl() {
			return {1.0};
		}

		static constexpr std::array<VoxelP0,1> dofs_on_elem_impl(const QuadElem_t el) {
			return {VoxelP0{el}};
		}
	};
}