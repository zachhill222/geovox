#pragma once

#include "mesh/keys/voxel_key.hpp"
#include "util/point.hpp"

#include <array>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//base DOF class to provide some utility methods.
	//evaulation, storage of support elments and such must be done in the derived classes.
	//each dof lives on some mesh feature. If any of the periodic BC flags are active,
	//then the constructor is changed to provide a single DOF regardless of which mesh feature
	//key was used. The "connonical" key is the one with the lower linear index.
	//each bit of BC sets one periodic BC boundary condition (lsb-x to msb-z)
	//for example 3 is 011 has X and Y periodic BC, while 6 is 110 has Y and Z
	template<VoxelKeyType Key_type, typename DERIVED>
	struct VoxelDOFBase
	{
		//type aliases to distinguish reference coordinates and geometric coordinate
		//purely for logical aid.
		using RefPoint_t = Point<3,double>;
		using GeoPoint_t = Point<3,double>;
		using Key_t      = Key_type;
		using QuadElem_t = VoxelElementKey<Key_t::MAX_DEPTH, Key_t::BC_FLAG>;
		using Hash       = Key_t::Hash;
		//store the logical key where this element lives
		//note that this key does not need to correspond to an active feature of the mesh
		Key_t key{};

		//import useful operations that should be forwarded to the feature key
		inline constexpr bool exists() const {return key.exists();}
		inline constexpr uint64_t depth() const {return key.depth();}
		inline constexpr bool is_valid() const {return key.is_valid();}
		inline constexpr bool operator==(const DERIVED other) const {return key==other.key;}
		inline constexpr bool operator!=(const DERIVED other) const {return key!=other.key;}
		inline constexpr bool operator<(const DERIVED other) const {return key<other.key;}
		inline constexpr bool operator>(const DERIVED other) const {return key>other.key;}
		inline constexpr uint64_t linear_index() const {return key.linear_index();}
		inline constexpr uint64_t compare_bits() const {return key.compare_bits();} //for the hash

		//record any periodic boundary conditions
		static constexpr uint64_t MAX_DEPTH = Key_t::MAX_DEPTH;
		static constexpr uint64_t BC = Key_t::BC_FLAG;
		static constexpr bool PX = BC&1;
		static constexpr bool PY = BC&2;
		static constexpr bool PZ = BC&4;
		static_assert(PX==Key_t::PX && PY==Key_t::PY && PZ==Key_t::PZ, "Check how voxel key boundary conditions are passed.");

		//check which bc is active. set the x/y/z bit to true if it is.
		//returns a number from 0 to 7.
		constexpr uint64_t bc_active() const {
			if constexpr (VoxelVertexKeyType<Key_t>) {
				uint64_t flag = 0;
				const uint64_t mi = uint64_t{1}<<key.depth(); //2^d elements per axis, also the max vertex index
				if constexpr (PX) {flag |= ((key.i()==0 || key.i()>=mi) ? 1 : 0);}
				if constexpr (PY) {flag |= ((key.j()==0 || key.j()>=mi) ? 2 : 0);}
				if constexpr (PZ) {flag |= ((key.k()==0 || key.k()>=mi) ? 4 : 0);}
				return flag;
			}
			else if constexpr (VoxelFaceKeyType<Key_t>) {
				const uint64_t aa = key.axis();
				const uint64_t mi = uint64_t{1} << key.depth(); //2^d elements, max index in the axis direction
				//only the BC in the axis direction can possibly be active
				if constexpr (PX) {if (aa==0) {return (key.i()==0 || key.i()>=mi) ? 1 : 0;}}
				if constexpr (PY) {if (aa==1) {return (key.j()==0 || key.j()>=mi) ? 2 : 0;}}
				if constexpr (PZ) {if (aa==2) {return (key.k()==0 || key.k()>=mi) ? 4 : 0;}}
			}
			
			//I don't think periodic BCs ever make sense if the DOF is defined on a single element
			return 0;
		}

		//constructors
		//keep a default so that std::vector<VoxelDOFBase>::resize() can be called.
		constexpr VoxelDOFBase() : key{} {}

		//standard constructor handles out-of bounds and periodic wrapping of dofs
		//this is very helpful when constructing child dofs
		constexpr VoxelDOFBase(Key_t k) : key(k) {
			//periodic part is handled in the Key_t constructor
			if (!key.is_valid()) {key._data_ = Key_t::DOES_NOT_EXIST;}
		}

		//convert compatible keys to the dof
		template<typename OtherKey_t> requires (VoxelEquivFeature<Key_t,OtherKey_t>)
		constexpr VoxelDOFBase(OtherKey_t ok) : VoxelDOFBase(static_cast<Key_t>(ok)) {}

		//xi are normalized coordinates to quad_elem
		//each component of xi is in [-1,1]
		//these are evaluated in the reference element.
		//in particular, the gradient will need to be scaled by the jacobian.
		//some DOF types might return a vector or matrix, so use auto here.
		//the return type must be specified in the Derived class.
		
		//the interface for any needed operations must follow the form
		// eval(VoxelElementKey support, const RefPoint_t& quad_point)
		//where the support element is a natural support element at the same depth as the key
		//and quad_point is a reference coordinate in [-1,1]^3
		//these values will be the result of calling quad_elem2support_elem.
		//it is the caller's responsibility to guarantee that support is actually a support element of the DOF

		//we will often need to check the basis function's support elements
		//the periodic templates should be set in the DERIVED class and passed to this class
		//so that the DERIVED class correctly gets the support elements.
		inline constexpr auto support() const {
			return static_cast<const DERIVED*>(this) -> support_impl();
		}

		constexpr int n_dof_per_elem() const {
			return static_cast<const DERIVED*>(this) -> N_DOF_PER_ELEM;
		}

		constexpr auto dofs_on_elem(const QuadElem_t el) const {
			return static_cast<const DERIVED*>(this) -> dofs_on_elem_impl(el);
		}

		//charms specific interface. only here to enforce conformity
		inline constexpr auto children() const {
			return static_cast<const DERIVED*>(this) -> children_impl();
		}

		constexpr auto children_coef() const {
			return static_cast<const DERIVED*>(this) -> children_coef_impl();
		}

		constexpr auto parents() const {
			return static_cast<const DERIVED*>(this) -> parents_impl();
		}

		constexpr auto parent_coefs() const {
			return static_cast<const DERIVED*>(this) -> parent_coefs_impl();
		}

		//project a reference point on a deeper quadrature element to a support element
		//use a vectorized version for quadrature (in the kernel class)
		constexpr void proj_to_support(QuadElem_t& spt, RefPoint_t& pt) const {
			assert(spt.is_valid());
			const uint64_t md = spt.depth(); //max starting depth, work up
			for (uint64_t d = md; d>key.depth(); --d) {
				//determine if the element is an even/odd child in each coordinate
				//and compute the increment to shift each coordinate
				const double dx = static_cast<double>(spt.i() & 1) - 0.5;
				const double dy = static_cast<double>(spt.j() & 1) - 0.5;
				const double dz = static_cast<double>(spt.k() & 1) - 0.5;

				pt[0] = 0.5*pt[0] + dx;
				pt[1] = 0.5*pt[1] + dy;
				pt[2] = 0.5*pt[2] + dz;

				spt = spt.parent();
			}
			assert(spt.depth()==key.depth());
		}
	};
}