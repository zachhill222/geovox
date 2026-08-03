#pragma once

#include "gutil.hpp"
#include "simd_keys/mesh/mesh_key_implementation.hpp"
#include "simd_keys/mesh/mesh_keys.hpp"
#include "simd_keys/dofs/utility.hpp"

#include <cstdint>


#define ASSERT_VALID_ELEMENT(el) GUTIL_ASSERT(Mesh3D::Exists(el) && Mesh3D::IsElement(el) && Mesh3D::IsCartesian(el))
#define ASSERT_VALID_VOXELQ1_DOF(dof) GUTIL_ASSERT(Mesh3D::Exists(dof) && Mesh3D::IsVertex(dof) && Mesh3D::IsCartesian(dof))


namespace GV {
namespace Keys {
namespace DOFS {
namespace LagrangeQ1 {


	///////////////////////////////////////////////////////////
	/// Standard Lagrange Q1 voxel elements defined on mesh vertices.
	/// Hierarchy relations are also implemented
	///
	/// Each shape function is defined on the [-1,1]^3 reference element
	/// If the reference coordinate of a vertex is (sx,sy,sz) in {-1,1}^3,
	/// then its shape function is N(X,Y,Z) = 0.125*(1+sx*X)*(1+sy*Y)*(1+sz*Z)
	/// for any (X,Y,Z) in [-1,1]^3.
	///
	/// We follow the local dof numbering on a support element
	/// is the same as the vertex numbering on the same element.
	///
	/// Note that some operations allow for a Period template.
	/// 
	/// The Period is a number 0bzyx, where the bit x,y,z sets
	/// the periodic nature of that axis. For example,
	/// Period = 6 = 0b110 is periodic in z and y but not x
	///
	/// Note that specifying a Period does not change the index of any
	/// feature, only which indices are feasible. This is to allow
	/// maximum compatibility between meshes using non-periodic features
	/// and DOFs using periodic features.
	///
	/// VoxelQ1 DOFs are encoded by (possibly periodic) vertices.
	/// Their evaluation on a given support element depends on their vertex
	/// number on the at element.
	///
	/// Any element must be decoded (i.e. cartesian form) for processing.
	///////////////////////////////////////////////////////////

	inline constexpr uint64_t N_DOF_PER_ELEM = 8;
	inline constexpr uint64_t N_SUPPORT_ELEM = 8;
	
	inline constexpr uint64_t N_CHILDREN = 27;
	inline constexpr uint64_t N_PARENTS  = 8;


	///////////////////////////////////////////////////////////
	/// Utility methods
	///////////////////////////////////////////////////////////
	GUTIL_DECLARE_SIMD()
	template<uint8_t Period> requires(Period<8)
	[[nodiscard]] inline constexpr uint8_t LocalDofNumber_SIMD(uint64_t spt, uint64_t dof) noexcept {
		ASSERT_VALID_VOXELQ1_DOF(dof)
		ASSERT_VALID_ELEMENT(spt)
		GUTIL_ASSERT(Mesh3D::IsValid<Period>(dof))
		return Mesh3D::GetLocalVertexNumberCartesian_SIMD<Period>(spt, dof);
	}

	GUTIL_DECLARE_SIMD()
	template<uint8_t Period> requires(Period<8)
	inline constexpr void GetDofsOnElement_SIMD(uint64_t spt, uint64_t* dof) noexcept {
		//dofs points to the start of N_DOF_PER_ELEM locations
		//if dof i is not valid due to periodicity, it will fail a Mesh3D::Exists(dof[i]) check
		GUTIL_ASSERT(dof)
		ASSERT_VALID_ELEMENT(spt)
		Mesh3D::GetVerticesOfElement_SIMD<Period>(spt,dof);
	}
	
	GUTIL_DECLARE_SIMD()
	template<uint8_t Period> requires(Period<8)
	inline constexpr void GetDofSupport_SIMD(uint64_t* spt, uint64_t dof) noexcept {
		//dofs points to the start of N_SUPPORT_ELEM locations
		GUTIL_ASSERT(spt)
		ASSERT_VALID_VOXELQ1_DOF(dof)
		GUTIL_ASSERT(Mesh3D::IsValid<Period>(dof))
		Mesh3D::GetElementsOfVertex_SIMD<Period>(dof, spt);
	}

	///////////////////////////////////////////////////////////
	/// Evaluation: standard single dof methods, convenience methods, performance evaluate all 8 dofs methods
	///////////////////////////////////////////////////////////
	template<typename T=double>
	inline constexpr void GetDofValueByLocalNumber(uint64_t dof, uint8_t local, T* val, const T* X, const T* Y, const T* Z, uint32_t N) noexcept {
		GUTIL_ASSERT(val && X && Y && Z && N>0);
		GUTIL_ASSERT(local<8);
		ASSERT_VALID_VOXELQ1_DOF(dof);

		const T sx = (local&1) ? T{1} : T{-1};
		const T sy = (local&2) ? T{1} : T{-1};
		const T sz = (local&4) ? T{1} : T{-1};

		GUTIL_SIMD()
		for (uint32_t i=0; i<N; ++i) {
			GUTIL_ASSERT(T{-1} <= X[i] && T{-1} <= Y[i] && T{-1} <= Z[i]);
			GUTIL_ASSERT(T{1}  >= X[i] && T{1}  >= Y[i] && T{1}  >= Z[i]);

			val[i] = T{0.125} * (T{1}+sx*X[i]) * (T{1}+sy*Y[i]) * (T{1}+sz*Z[i]);
		}
	}

	template<typename T=double>
	inline constexpr void GetDofGradientByLocalNumber(uint64_t dof, uint8_t local, T* val_x, T* val_y, T* val_z, const T* X, const T* Y, const T* Z, uint32_t N) noexcept {
		GUTIL_ASSERT(val_x && val_y && val_z && X && Y && Z && N>0);
		GUTIL_ASSERT(local<8);
		ASSERT_VALID_VOXELQ1_DOF(dof);

		const T sx = (local&1) ? T{1} : T{-1};
		const T sy = (local&2) ? T{1} : T{-1};
		const T sz = (local&4) ? T{1} : T{-1};

		GUTIL_SIMD()
		for (uint32_t i=0; i<N; ++i) {
			GUTIL_ASSERT(T{-1} <= X[i] && T{-1} <= Y[i] && T{-1} <= Z[i]);
			GUTIL_ASSERT(T{1}  >= X[i] && T{1}  >= Y[i] && T{1}  >= Z[i]);

			//regular/non-derivative portions of the evaluation
			const T rx = T{1}+sx*X[i];
			const T ry = T{1}+sy*Y[i];
			const T rz = T{1}+sz*Z[i];

			val_x[i]   = T{0.125} * sx * ry * rz;
			val_y[i]   = T{0.125} * rx * sy * rz;
			val_z[i]   = T{0.125} * rx * ry * sz;
		}
	}

	template<uint8_t Period, typename PointContainer> requires(Period<8)
	[[nodiscard]] inline constexpr typename PointContainer::value_type GetDofValue(uint64_t spt, uint64_t dof, const PointContainer& pt) noexcept {
		typename PointContainer::value_type val;
		GetDofValueByLocalNumber(dof, LocalDofNumber_SIMD<Period>(spt,dof), &val, &pt[0], &pt[1], &pt[2], 1);
		return val;
	}

	template<uint8_t Period, typename PointContainer> requires(Period<8)
	[[nodiscard]] inline constexpr PointContainer GetDofGradient(uint64_t spt, uint64_t dof, const PointContainer& pt) noexcept {
		PointContainer grad{0,0,0};
		GetDofGradientByLocalNumber(dof, LocalDofNumber_SIMD<Period>(spt,dof), &grad[0], &grad[1], &grad[2], &pt[0], &pt[1], &pt[2], 1);
		return grad;
	}


	///////////////////////////////////////////////////////////
	/// Hierarchy
	///////////////////////////////////////////////////////////

	//coefficeints of children are 0.5^(di+dj+dk) where (di,dj,dk) in {-1,0,1}^3
	//is the index offset from the dof child vertex to the vertex of the child dof
	inline constexpr std::array<double,N_CHILDREN> ChildrenCoefficients = {
		//bottom plane (k=-1)
		0.125, 0.25, 0.125,
		0.25,  0.5,  0.25,
		0.125, 0.25, 0.125,

		//middle plane (k=0)
		0.25,  0.5,  0.25,
		0.5,   1.0,  0.5,		//note the dof at the child vertex has coef 1
		0.25,  0.5,  0.25,

		//top plane (k=1)
		0.125, 0.25, 0.125,
		0.25,  0.5,  0.25,
		0.125, 0.25, 0.125
	};

	GUTIL_DECLARE_SIMD()
	template<typename T=double>
	[[nodiscard]] inline constexpr T GetChildCoef(uint8_t idx) noexcept {
		GUTIL_ASSERT(idx<27);
		return T{ChildrenCoefficients[idx]};
	}

	GUTIL_DECLARE_SIMD()
	template<uint8_t Period> requires(Period<8)
	inline constexpr void GetDofChildren_SIMD(uint64_t dof, uint64_t* children) noexcept {
		//children must point to the first element of N_CHILDREN locations
		//any children that don't exist or wrap due to periodicity will be handled.
		//similarly, if the depth is too large, the children will be marked as 'does not exist'
		ASSERT_VALID_VOXELQ1_DOF(dof)
		GUTIL_ASSERT(Mesh3D::IsValid<Period>(dof))
		
		const uint64_t dd = Mesh3D::Depth(dof) + 1;
		const uint64_t ii = 2*Mesh3D::IndexI_SIMD(dof); const uint64_t im1=ii-1; const uint64_t ip1=ii+1;
		const uint64_t jj = 2*Mesh3D::IndexJ_SIMD(dof); const uint64_t jm1=jj-1; const uint64_t jp1=jj+1;
		const uint64_t kk = 2*Mesh3D::IndexK_SIMD(dof); const uint64_t km1=kk-1; const uint64_t kp1=kk+1;

		//the MakeVertex<Period> factory handles validating and correcting the period

		//bottom plane (k=-1)
		children[0]  = Mesh3D::MakeVertex<Period>(dd, im1, jm1, km1);
		children[1]  = Mesh3D::MakeVertex<Period>(dd, ii , jm1, km1);
		children[2]  = Mesh3D::MakeVertex<Period>(dd, ip1, jm1, km1);
		children[3]  = Mesh3D::MakeVertex<Period>(dd, im1, jj,  km1);
		children[4]  = Mesh3D::MakeVertex<Period>(dd, ii , jj,  km1);
		children[5]  = Mesh3D::MakeVertex<Period>(dd, ip1, jj,  km1);
		children[6]  = Mesh3D::MakeVertex<Period>(dd, im1, jp1, km1);
		children[7]  = Mesh3D::MakeVertex<Period>(dd, ii , jp1, km1);
		children[8]  = Mesh3D::MakeVertex<Period>(dd, ip1, jp1, km1);

		//middle plane (k=0)
		children[9]  = Mesh3D::MakeVertex<Period>(dd, im1, jm1, kk  );
		children[10] = Mesh3D::MakeVertex<Period>(dd, ii , jm1, kk  );
		children[11] = Mesh3D::MakeVertex<Period>(dd, ip1, jm1, kk  );
		children[12] = Mesh3D::MakeVertex<Period>(dd, im1, jj,  kk  );
		children[13] = Mesh3D::MakeVertex<Period>(dd, ii , jj,  kk  );
		children[14] = Mesh3D::MakeVertex<Period>(dd, ip1, jj,  kk  );
		children[15] = Mesh3D::MakeVertex<Period>(dd, im1, jp1, kk  );
		children[16] = Mesh3D::MakeVertex<Period>(dd, ii , jp1, kk  );
		children[17] = Mesh3D::MakeVertex<Period>(dd, ip1, jp1, kk  );

		//top plane (k=1)
		children[18] = Mesh3D::MakeVertex<Period>(dd, im1, jm1, kp1);
		children[19] = Mesh3D::MakeVertex<Period>(dd, ii , jm1, kp1);
		children[20] = Mesh3D::MakeVertex<Period>(dd, ip1, jm1, kp1);
		children[21] = Mesh3D::MakeVertex<Period>(dd, im1, jj,  kp1);
		children[22] = Mesh3D::MakeVertex<Period>(dd, ii , jj,  kp1);
		children[23] = Mesh3D::MakeVertex<Period>(dd, ip1, jj,  kp1);
		children[24] = Mesh3D::MakeVertex<Period>(dd, im1, jp1, kp1);
		children[25] = Mesh3D::MakeVertex<Period>(dd, ii , jp1, kp1);
		children[26] = Mesh3D::MakeVertex<Period>(dd, ip1, jp1, kp1);
	}

	//the coeficients of the parents are all 1/n_parents_that_exist
	//where n_parents_that_exist depends on the feature that the projection
	//of the dof vertex lands on when going up a hierarchy level.
	//see GetDofParents_SIMD below.
	GUTIL_DECLARE_SIMD()
	template<typename T=double, uint8_t Period> requires(Period<8)
	[[nodiscard]] inline constexpr T GetParentCoef(uint64_t dof) noexcept {
		GUTIL_ASSERT(Mesh3D::Depth(dof)>0)
		ASSERT_VALID_VOXELQ1_DOF(dof)
		GUTIL_ASSERT(Mesh3D::IsValid<Period>(dof))
		const uint64_t ii = Mesh3D::IndexI_SIMD(dof);
		const uint64_t jj = Mesh3D::IndexJ_SIMD(dof);
		const uint64_t kk = Mesh3D::IndexK_SIMD(dof);
		int bi = ii&1, bj = jj&1, bk = kk&1;

		//at parent depth 0, a periodic axis wraps every position onto the same
		//unique parent, so the dof genuinely shares that feature regardless of
		//its raw parity -- same override as before, still needed
		if (Mesh3D::Depth(dof)==1) {
			if constexpr (Period&0b001) {bi=0;}
			if constexpr (Period&0b010) {bj=0;}
			if constexpr (Period&0b100) {bk=0;}
		}

		//only the dof that coincides exactly with the parent's own position
		//(all axes even) contributes anything -- everything else is a genuinely
		//different geometric point and must contribute 0, so that refine->unrefine
		//is lossless for a constant field
		return (bi==0 && bj==0 && bk==0) ? T{1} : T{0};
	}



	GUTIL_DECLARE_SIMD()
	template<uint8_t Period> requires(Period<8)
	inline constexpr void GetDofParents_SIMD(uint64_t dof, uint64_t* parents) noexcept {
		//a parent of the dof is any dof at one depth higher whose support has a positive measure
		//intersection with the dof support.

		//parents must point to the first element of N_PARENTS locations
		//any parents that don't exist or wrap due to periodicity will be handled.
		//similarly, if the depth is 0, the parents will be marked as 'does not exist'
		ASSERT_VALID_VOXELQ1_DOF(dof)
		GUTIL_ASSERT(Mesh3D::IsValid<Period>(dof))
		
		//the dof is the center of a virtual (it may not be correcly alligned) element at depth dd-1
		//the parent dofs are the local dofs of this virtual element, we use bitwise computations similar to
		//Mesh3D::GetLocalVertexNumberCartesian_SIMD to compute this, but we cannot use the standard method
		//as the virtual element most likely does not align to the encoded element hierarchy

		//the number of parents depends on the relative location of the child dof vertex
		//relative to the the elements at depth dd-1 (i.e, is it in the center of an element, on a face, on an edge, on a vertex)
		//
		// vertex -> 1 parent
		// edge   -> 2 parents
		// face   -> 4 parents
		// element-> 8 parents

		const uint64_t dd = Mesh3D::Depth(dof) - 1;
		const uint64_t ii = Mesh3D::IndexI_SIMD(dof);
		const uint64_t jj = Mesh3D::IndexJ_SIMD(dof);
		const uint64_t kk = Mesh3D::IndexK_SIMD(dof);

		uint64_t bi  = ii&1,   bj  = jj&1,   bk  = kk&1;	//capture pairity of dof index
		uint64_t ip0 = ii>>1,  jp0 = jj>>1,  kp0 = kk>>1;	//get index of the 'low' parent
		uint64_t ip1 = ip0+bi, jp1 = jp0+bj, kp1 = kp0+bk;	//get the index of the 'high' parent

		//for periodic axes at depth 0 the mask needs to be updated
		//otherwise, the periodic factory will double count the number of parents
		//in each axis.
		if constexpr (Period!=0) {
			if (dd==0) {
				if constexpr (Period&0b001) { bi=0; }
				if constexpr (Period&0b010) { bj=0; }
				if constexpr (Period&0b100) { bk=0; }
			}
		}



		//invert and mask the pairity bits. Parent p in [0,8) exists if p&mask == 0
		const uint64_t mask = ~(0b111 & ((bk<<2) | (bj<<1) | (bi)));

		//below
		parents[0] = 				 Mesh3D::MakeVertex<Period>(dd, ip0, jp0, kp0);	//0&mask==0 always
		parents[1] = ((1&mask)==0) ? Mesh3D::MakeVertex<Period>(dd, ip1, jp0, kp0) : 0;
		parents[2] = ((2&mask)==0) ? Mesh3D::MakeVertex<Period>(dd, ip0, jp1, kp0) : 0;
		parents[3] = ((3&mask)==0) ? Mesh3D::MakeVertex<Period>(dd, ip1, jp1, kp0) : 0;
		//above
		parents[4] = ((4&mask)==0) ? Mesh3D::MakeVertex<Period>(dd, ip0, jp0, kp1) : 0;
		parents[5] = ((5&mask)==0) ? Mesh3D::MakeVertex<Period>(dd, ip1, jp0, kp1) : 0;
		parents[6] = ((6&mask)==0) ? Mesh3D::MakeVertex<Period>(dd, ip0, jp1, kp1) : 0;
		parents[7] = ((7&mask)==0) ? Mesh3D::MakeVertex<Period>(dd, ip1, jp1, kp1) : 0;
	}






}}}}
