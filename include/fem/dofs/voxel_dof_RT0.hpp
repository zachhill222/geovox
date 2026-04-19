#pragma once

#include "fem/dofs/voxel_dof_base.hpp"
#include "mesh/keys/voxel_key.hpp"

#include <array>
#include <cstdint>
#include <omp.h>

namespace GV
{
	template<VoxelFaceKeyType Key_type>
	struct VoxelRT0 : public VoxelDOFBase<Key_type,VoxelRT0<Key_type>>
	{
		//get types from the Base class
		using Base = VoxelDOFBase<Key_type,VoxelRT0<Key_type>>;
		using RefPoint_t = typename Base::RefPoint_t;
		using GeoPoint_t = typename Base::GeoPoint_t;
		using Key_t      = typename Base::Key_t;
		using QuadElem_t = typename Base::QuadElem_t;

		//constants
		static constexpr uint64_t N_CHILDREN     = 12;
		static constexpr uint64_t N_DOF_PER_ELEM = 6;
		static constexpr uint64_t N_SUPPORT_ELEM = 2;
		static constexpr uint64_t N_PARENTS      = 2;

		//access the data
		using Base::key;

		//use the base constructors
		using Base::Base;

		//evaluate normal component. this is the same as the flux through the face.
		//use the vectorized version. this is only for debugging.
		constexpr double flux(QuadElem_t support_elem, const RefPoint_t& xi) const {
			//similar to Q1, but only evaluate the coordinate in the normal direction
			//if the dof is normal to the x-axis and on the low side,
			//then we should return 1 when xi[0]=-1 and 0 when xi[0]=1.
			//similar reasoning applies to the other faces.
			const uint64_t a = key.axis();
			std::array<double,1> vl;
			std::array<double,1> nx{xi[a]};
			flux(vl, support_elem, nx);
			return vl[0];
		}

		//use the vectorized flux. this is only for debugging.
		constexpr RefPoint_t eval(QuadElem_t support_elem, const RefPoint_t& xi) const {
			const uint64_t a = key.axis();
			RefPoint_t result{0.0, 0.0, 0.0};

			result[a] = flux(support_elem, xi); //evaluate normal component
			return result;
		}

		//use the vectorized version. this is only for debugging.
		constexpr double div(QuadElem_t support_elem, const RefPoint_t& xi) const {
			const uint64_t a = key.axis();
			const bool low_face = 	(a==0) ? (key.i() == support_elem.i()) :
									(a==1) ? (key.j() == support_elem.j()) :
											 (key.k() == support_elem.k());
			return low_face ? -0.5 : 0.5;
		}

		//vectorized flux, nx is the normal distance to the face
		template<int N> requires (N>0)
		void flux(	std::array<double,N>& vl, QuadElem_t el, const std::array<double,N>& nx) const {
			

			const uint64_t a    = key.axis();
			const bool low_face = 	(a==0) ? (key.i() == el.i()) :
									(a==1) ? (key.j() == el.j()) :
											 (key.k() == el.k());
			const bool s = low_face ? -0.5 : 0.5;
			#pragma omp simd
			for (int q=0; q<N; ++q) {
				vl[q] = 0.5 + s*nx[q];
			}
		}

		//vectorized div
		template<int N> requires (N>0)
		void div(std::array<double,N>& vl, QuadElem_t el) const {
			
			const uint64_t a    = key.axis();
			const bool low_face = 	(a==0) ? (key.i() == el.i()) :
									(a==1) ? (key.j() == el.j()) :
											 (key.k() == el.k());
			
			if (low_face) {vl.fill(-0.5);}
			else {vl.fill(0.5);}
		}

		constexpr std::array<VoxelRT0,12> children_impl() const {
			const uint64_t aa=key.axis(), ii=2*key.i(), jj=2*key.j(), kk=2*key.k(), dd=key.depth()+1;
			assert(dd<key.MAX_DEPTH);

			switch (aa) {
			case 0:
				return {
					VoxelRT0{Key_t{aa,dd, ii-1, jj,   kk  }},
					VoxelRT0{Key_t{aa,dd, ii-1, jj+1, kk  }},
					VoxelRT0{Key_t{aa,dd, ii-1, jj,   kk+1}},
					VoxelRT0{Key_t{aa,dd, ii-1, jj+1, kk+1}},

					VoxelRT0{Key_t{aa,dd, ii  , jj,   kk  }},
					VoxelRT0{Key_t{aa,dd, ii  , jj+1, kk  }},
					VoxelRT0{Key_t{aa,dd, ii  , jj,   kk+1}},
					VoxelRT0{Key_t{aa,dd, ii  , jj+1, kk+1}},

					VoxelRT0{Key_t{aa,dd, ii+1, jj,   kk  }},
					VoxelRT0{Key_t{aa,dd, ii+1, jj+1, kk  }},
					VoxelRT0{Key_t{aa,dd, ii+1, jj,   kk+1}},
					VoxelRT0{Key_t{aa,dd, ii+1, jj+1, kk+1}}
				};
			
			case 1:
				return {
					VoxelRT0{Key_t{aa,dd, ii,   jj-1, kk  }},
					VoxelRT0{Key_t{aa,dd, ii+1, jj-1, kk  }},
					VoxelRT0{Key_t{aa,dd, ii,   jj-1, kk+1}},
					VoxelRT0{Key_t{aa,dd, ii+1, jj-1, kk+1}},

					VoxelRT0{Key_t{aa,dd, ii,   jj  , kk  }},
					VoxelRT0{Key_t{aa,dd, ii+1, jj  , kk  }},
					VoxelRT0{Key_t{aa,dd, ii,   jj  , kk+1}},
					VoxelRT0{Key_t{aa,dd, ii+1, jj  , kk+1}},

					VoxelRT0{Key_t{aa,dd, ii,   jj+1, kk  }},
					VoxelRT0{Key_t{aa,dd, ii+1, jj+1, kk  }},
					VoxelRT0{Key_t{aa,dd, ii,   jj+1, kk+1}},
					VoxelRT0{Key_t{aa,dd, ii+1, jj+1, kk+1}}
				};
			
			case 2:
				return {
					VoxelRT0{Key_t{aa,dd, ii,   jj  , kk-1}},
					VoxelRT0{Key_t{aa,dd, ii+1, jj  , kk-1}},
					VoxelRT0{Key_t{aa,dd, ii,   jj+1, kk-1}},
					VoxelRT0{Key_t{aa,dd, ii+1, jj+1, kk-1}},

					VoxelRT0{Key_t{aa,dd, ii,   jj  , kk  }},
					VoxelRT0{Key_t{aa,dd, ii+1, jj  , kk  }},
					VoxelRT0{Key_t{aa,dd, ii,   jj+1, kk  }},
					VoxelRT0{Key_t{aa,dd, ii+1, jj+1, kk  }},

					VoxelRT0{Key_t{aa,dd, ii,   jj  , kk+1}},
					VoxelRT0{Key_t{aa,dd, ii+1, jj  , kk+1}},
					VoxelRT0{Key_t{aa,dd, ii,   jj+1, kk+1}},
					VoxelRT0{Key_t{aa,dd, ii+1, jj+1, kk+1}}
				};
			}
		}

		static constexpr std::array<double,12> children_coef_impl() {
			return {
				0.5, 0.5, 0.5, 0.5,
				1.0, 1.0, 1.0, 1.0,
				0.5, 0.5, 0.5, 0.5
			};
		}


		constexpr std::array<VoxelRT0,N_PARENTS> parents_impl() const {
			const uint64_t dd = key.depth() -1; //underflow if 0
			if (dd>Key_t::MAX_DEPTH) {return {};} //no parents when depth is 0

			//get parent indices (lower left)
			const uint64_t aa=key.axis(), ip=key.i()>>1, jp=key.j()>>1, kp=key.k()>>1;
			
			//check if the index in the axis direction is even/odd
			//even indices have on parent, odd indices have 2
			const uint64_t pr = key.index(aa)&1;
			if (pr) {
				switch (aa) {
				case 0: return {VoxelRT0{Key_t{aa,dd,ip,jp,kp}}, VoxelRT0{Key_t{aa,dd,ip+1,jp,  kp  }}};
				case 1: return {VoxelRT0{Key_t{aa,dd,ip,jp,kp}}, VoxelRT0{Key_t{aa,dd,ip,  jp+1,kp  }}};
				case 2: return {VoxelRT0{Key_t{aa,dd,ip,jp,kp}}, VoxelRT0{Key_t{aa,dd,ip,  jp,  kp+1}}};
				default: assert(false); return {};
				}
			}
			else {
				//one parent
				return {VoxelRT0{Key_t{aa,dd,ip,jp,kp}}, VoxelRT0{}};
			}
		}

		constexpr std::array<double,N_PARENTS> parent_coefs_impl() const {
			if (key.index(key.axis())&1) {return {0.5, 0.5};} //faces with odd axis indices have two parents
			return {1.0, 1.0};
		}

		static constexpr std::array<VoxelRT0,6> dofs_on_elem(const QuadElem_t el) {
			std::array<VoxelRT0,6> dofs;
			for (int f=0; f<6; ++f) {dofs[f] = VoxelRT0{el.face(f)};}
			return dofs;
		}
	};
}