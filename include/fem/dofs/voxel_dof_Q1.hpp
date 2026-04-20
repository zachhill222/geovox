#pragma once

#include "fem/dofs/voxel_dof_base.hpp"
#include "mesh/keys/voxel_key.hpp"

#include <array>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//Q1 DOFs
	template<VoxelVertexKeyType Key_type>
	struct VoxelQ1 : public VoxelDOFBase<Key_type, VoxelQ1<Key_type>>
	{
		//get types from the Base class
		using Base = VoxelDOFBase<Key_type, VoxelQ1<Key_type>>;
		using RefPoint_t = typename Base::RefPoint_t;
		using GeoPoint_t = typename Base::GeoPoint_t;
		using Key_t      = typename Base::Key_t;
		using QuadElem_t = typename Base::QuadElem_t;

		//constants
		static constexpr uint64_t N_CHILDREN     = 27;
		static constexpr uint64_t N_DOF_PER_ELEM = 8;
		static constexpr uint64_t N_SUPPORT_ELEM = 8;
		static constexpr uint64_t N_PARENTS      = 8;

		using Base::PX;
		using Base::PY;
		using Base::PZ;

		//access the data
		using Base::key;

		//use the base constructors
		using Base::Base;

		//evaluate the basis function (convenience method)
		double eval(QuadElem_t support_elem, const RefPoint_t& xi) const {
			std::array<double,1> val;
			std::array<double,1> x{xi[0]};
			std::array<double,1> y{xi[1]};
			std::array<double,1> z{xi[2]};

			eval<1>(val, support_elem, x, y, z);
			return val[0];
		}


		//compute the gradient of the basis function on the reference coordinate
		//the caller must scale the gradient to the mesh domain by
		//multiplying by the inverse transpose jacobian (i.e., divide by half the diagonal of the geometric support element)
		RefPoint_t grad(QuadElem_t support_elem, const RefPoint_t& xi) const {
			std::array<double,1> gx, gy, gz;
			std::array<double,1> x{xi[0]};
			std::array<double,1> y{xi[1]};
			std::array<double,1> z{xi[2]};

			grad(gx, gy, gz, support_elem, x, y, z);
			return RefPoint_t{gx[0], gy[0], gz[0]};
		}

		//vectorized evaluation
		template<uint64_t N, typename Elem_type> requires (N>0 && VoxelEquivFeature<Elem_type,QuadElem_t>)
		void eval(	std::array<double,N>&		vl, //values 
					Elem_type                   el, //support element
					const std::array<double,N>& qx, //reference/quadrature points
					const std::array<double,N>& qy, 
					const std::array<double,N>& qz) const {
			//check that the index logic is correct
			assert(el.is_valid());
			assert(key.depth() == el.depth());
			assert(key.i() - el.i() <= 1);
			assert(key.j() - el.j() <= 1);
			assert(key.k() - el.k() <= 1);
			
			const bool bx = static_cast<bool>(key.i() - el.i());
			const bool by = static_cast<bool>(key.j() - el.j());
			const bool bz = static_cast<bool>(key.k() - el.k());

			const double sx = bx ? 1.0 : -1.0;
			const double sy = by ? 1.0 : -1.0;
			const double sz = bz ? 1.0 : -1.0;

			#pragma omp simd
			for (uint64_t q=0; q<N; ++q) {
				vl[q] = 0.125 * (1.0+sx*qx[q]) * (1.0+sy*qy[q]) * (1.0+sz*qz[q]);
			}
		}

		//vectorized grad
		template<uint64_t N, typename Elem_type> requires (N>0 && VoxelEquivFeature<Elem_type,QuadElem_t>)
		void grad(	std::array<double,N>&		gx, //gradient result
					std::array<double,N>& 		gy, 
					std::array<double,N>& 		gz, 
					Elem_type  	 				el, //support element
					const std::array<double,N>& qx, //reference/quadratrue points
					const std::array<double,N>& qy, 
					const std::array<double,N>& qz) const {

			//check that the index logic is correct
			assert(el.is_valid());
			assert(key.i() - el.i() <= 1);
			assert(key.j() - el.j() <= 1);
			assert(key.k() - el.k() <= 1);
			
			const bool bx = static_cast<bool>(key.i() - el.i());
			const bool by = static_cast<bool>(key.j() - el.j());
			const bool bz = static_cast<bool>(key.k() - el.k());

			const double sx = bx ? 1.0 : -1.0;
			const double sy = by ? 1.0 : -1.0;
			const double sz = bz ? 1.0 : -1.0;

			#pragma omp simd
			for (size_t q=0; q<N; ++q) {
				const double fx = 1.0+sx*qx[q];
				const double fy = 1.0+sy*qy[q];
				const double fz = 1.0+sz*qz[q];

				gx[q] = 0.125 * sx * fy * fz;
				gy[q] = 0.125 * fx * sy * fz;
				gz[q] = 0.125 * fx * fy * sz;
			}
		}

		//get the support elements
		constexpr std::array<QuadElem_t, 8> support_impl() const {
			//the key constructor will handle any periodic wrapping
			std::array<QuadElem_t,8> spt = key.elements();

			//handle non-periodic out of bounds
			for (auto& el : spt) {
				if (!el.is_valid()) {el = QuadElem_t{};}
			}

			return spt;
		}

		constexpr std::array<VoxelQ1,N_CHILDREN> children_impl() const {
			const uint64_t ii=2*key.i(), jj=2*key.j(), kk=2*key.k(), dd=key.depth()+1;
			assert(dd<Key_t::MAX_DEPTH);
			return {
				//bottom plane (k=-1)
				VoxelQ1{Key_t{dd, ii-1, jj-1, kk-1}},
				VoxelQ1{Key_t{dd, ii ,  jj-1, kk-1}},
				VoxelQ1{Key_t{dd, ii+1, jj-1, kk-1}},
				VoxelQ1{Key_t{dd, ii-1, jj,   kk-1}},
				VoxelQ1{Key_t{dd, ii ,  jj,   kk-1}},
				VoxelQ1{Key_t{dd, ii+1, jj,   kk-1}},
				VoxelQ1{Key_t{dd, ii-1, jj+1, kk-1}},
				VoxelQ1{Key_t{dd, ii ,  jj+1, kk-1}},
				VoxelQ1{Key_t{dd, ii+1, jj+1, kk-1}},

				//middle plane (k=0)
				VoxelQ1{Key_t{dd, ii-1, jj-1, kk }},
				VoxelQ1{Key_t{dd, ii ,  jj-1, kk }},
				VoxelQ1{Key_t{dd, ii+1, jj-1, kk }},
				VoxelQ1{Key_t{dd, ii-1, jj,   kk }},
				VoxelQ1{Key_t{dd, ii ,  jj,   kk }},
				VoxelQ1{Key_t{dd, ii+1, jj,   kk }},
				VoxelQ1{Key_t{dd, ii-1, jj+1, kk }},
				VoxelQ1{Key_t{dd, ii ,  jj+1, kk }},
				VoxelQ1{Key_t{dd, ii+1, jj+1, kk }},

				//top plane (k=1)
				VoxelQ1{Key_t{dd, ii-1, jj-1, kk+1}},
				VoxelQ1{Key_t{dd, ii ,  jj-1, kk+1}},
				VoxelQ1{Key_t{dd, ii+1, jj-1, kk+1}},
				VoxelQ1{Key_t{dd, ii-1, jj,   kk+1}},
				VoxelQ1{Key_t{dd, ii ,  jj,   kk+1}},
				VoxelQ1{Key_t{dd, ii+1, jj,   kk+1}},
				VoxelQ1{Key_t{dd, ii-1, jj+1, kk+1}},
				VoxelQ1{Key_t{dd, ii ,  jj+1, kk+1}},
				VoxelQ1{Key_t{dd, ii+1, jj+1, kk+1}},
			};
		}

		static constexpr std::array<double,N_CHILDREN> children_coef_impl() {
			return {
				//bottom plane (k=-1)
				0.125, 0.25, 0.125,
				0.25,  0.5,  0.25,
				0.125, 0.25, 0.125,

				//middle plane (k=0)
				0.25,  0.5,  0.25,
				0.5,   1.0,  0.5,
				0.25,  0.5,  0.25,

				//top plane (k=1)
				0.125, 0.25, 0.125,
				0.25,  0.5,  0.25,
				0.125, 0.25, 0.125
			};
		}

		constexpr std::array<VoxelQ1,N_PARENTS> parents_impl() const {
			std::array<VoxelQ1,N_PARENTS> parents;
			parents.fill(VoxelQ1{});

			const uint64_t dd = key.depth() -1; //underflow if 0
			if (dd>Key_t::MAX_DEPTH) {return parents;} //no parents when depth is 0

			const uint64_t ii=key.i(), jj=key.j(), kk=key.k(); //bottom lower left

			//odd indices contribute two parents in that direction
			//there are 2^(bi+bj+bk) parents (except at depth 0 where there are none)
			const uint64_t bi=ii&1, bj=jj&1, bk=kk&1;
			const uint64_t pr = bi | (bj<<1) | (bk<<2);

			//get base index of the parents (lowest bottom left)
			const uint64_t ip=ii>>1;
			const uint64_t jp=jj>>1;
			const uint64_t kp=kk>>1;

			//branch by the pr : bk|bj|bi
			//note that the coefficients depend on the pairity
			switch (pr) {
			case 0: //000
				parents[0] = VoxelQ1{Key_t{dd, ip,   jp,   kp  }}; break;
			case 1: //001
				parents[0] = VoxelQ1{Key_t{dd, ip,   jp,   kp  }};
				parents[1] = VoxelQ1{Key_t{dd, ip+1, jp,   kp  }}; break;
			case 2: //010
				parents[0] = VoxelQ1{Key_t{dd, ip,   jp,   kp  }};
				parents[1] = VoxelQ1{Key_t{dd, ip,   jp+1, kp  }}; break;
			case 3: //011
				parents[0] = VoxelQ1{Key_t{dd, ip,   jp,   kp  }};
				parents[1] = VoxelQ1{Key_t{dd, ip+1, jp,   kp  }};
				parents[2] = VoxelQ1{Key_t{dd, ip,   jp+1, kp  }};
				parents[3] = VoxelQ1{Key_t{dd, ip+1, jp+1, kp  }}; break;
			case 4: //100
				parents[0] = VoxelQ1{Key_t{dd, ip,   jp,   kp  }};
				parents[1] = VoxelQ1{Key_t{dd, ip,   jp,   kp+1}}; break;
			case 5: //101
				parents[0] = VoxelQ1{Key_t{dd, ip,   jp,   kp  }};
				parents[1] = VoxelQ1{Key_t{dd, ip+1, jp,   kp  }};
				parents[2] = VoxelQ1{Key_t{dd, ip,   jp,   kp+1}};
				parents[3] = VoxelQ1{Key_t{dd, ip+1, jp,   kp+1}}; break;
			case 6: //110
				parents[0] = VoxelQ1{Key_t{dd, ip,   jp,   kp  }};
				parents[1] = VoxelQ1{Key_t{dd, ip,   jp+1, kp  }};
				parents[2] = VoxelQ1{Key_t{dd, ip,   jp,   kp+1}};
				parents[3] = VoxelQ1{Key_t{dd, ip,   jp+1, kp+1}}; break;
			case 7: //111
				parents[0] = VoxelQ1{Key_t{dd, ip,   jp,   kp  }};
				parents[1] = VoxelQ1{Key_t{dd, ip+1, jp,   kp  }};
				parents[2] = VoxelQ1{Key_t{dd, ip,   jp+1, kp  }};
				parents[3] = VoxelQ1{Key_t{dd, ip+1, jp+1, kp  }};
				parents[4] = VoxelQ1{Key_t{dd, ip,   jp,   kp+1}};
				parents[5] = VoxelQ1{Key_t{dd, ip+1, jp,   kp+1}};
				parents[6] = VoxelQ1{Key_t{dd, ip,   jp+1, kp+1}};
				parents[7] = VoxelQ1{Key_t{dd, ip+1, jp+1, kp+1}}; break;
			}
			return parents;
		}


		constexpr std::array<double,N_PARENTS> parent_coefs_impl() const {
			std::array<double,N_PARENTS> coefs{};

			const uint64_t dd = key.depth() -1; //underflow if 0
			if (dd>Key_t::MAX_DEPTH) {return coefs;} //no parents when depth is 0

			const double pc[4] {1.0, 0.5, 0.25, 0.125}; //possible coefficients
			const uint64_t ii=key.i(), jj=key.j(), kk=key.k(); //bottom lower left

			//odd indices contribute two parents in that direction
			//there are 2^(bi+bj+bk) parents (except at depth 0 where there are none)
			const uint64_t bi=ii&1, bj=jj&1, bk=kk&1;
			const uint64_t idx = bi+bj+bk; //the shift is also the index into pc
			const uint64_t np  = uint64_t{1} << idx; //number of parents

			//each parent evaluates to the same quantity at a given child function
			//also all coefficients must sum to 1
			std::fill_n(coefs.begin(), np, pc[idx]);
			return coefs;
		}

		
		template<typename OtherElem_t> requires (VoxelEquivFeature<OtherElem_t,QuadElem_t>)
		static constexpr std::array<VoxelQ1,N_DOF_PER_ELEM> dofs_on_elem(const OtherElem_t el) {
			const auto keys = static_cast<QuadElem_t>(el).vertices();
			std::array<VoxelQ1,N_DOF_PER_ELEM> dofs;

			for (int v=0; v<8; ++v) {
				dofs[v] = keys[v].is_valid() ? VoxelQ1{keys[v]} : VoxelQ1{};
			}
			return dofs;
		}
	};
}