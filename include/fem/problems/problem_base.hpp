#pragma once

#include "mesh/voxel_mesh.hpp"					//the mesh class
#include "fem/dofs/voxel_dofs.hpp"				//all dofs on voxels
#include "fem/handlers/dofhandler_charms.hpp"	//dofhandler to store dof hierarchies. need one handler per dof type
#include "fem/numerics/kernel.hpp"				//a type to organize integrating linear and bilinear forms. contains references only and can have one per thread.
#include "fem/handlers/bc_handler.hpp"				//a class to apply essential bcs
#include "fem/forms/forms.hpp"					//base classes to define the problem. pass these to the kernel.


#include <Eigen/SparseCore>

namespace GV
{
	//a base class to help with defining problems
	template <VoxelMeshType Mesh_type>
	struct BaseProblem
	{
		//define convenient type aliases for the mesh
		using Mesh_t = Mesh_type;
		using Elem_t = typename Mesh_t::VoxelElement;
		using Face_t = typename Mesh_t::VoxelFace;
		using Vert_t = typename Mesh_t::VoxelVertex;

		//define alias for dofhandlers
		template<typename DOF_TYPE>
		using DofHandler_T = DofHandlerCharms<Mesh_t,DOF_TYPE>;

		//bring the BChandler into this scope. templated on the DOF type.
		// using BCHandler;

		//The kernel type needs to be defined in terms of the required linear and bilinear forms
		//its type should look like 
		//
		//	using Kernel_t = Kernel<4, TypeList<Bilinear_t0, Bilinear_t1,...>, TypeList<Linear_t0, Linear_t1,...>>
		// 
		//where the 4 means four quadrature points per axis (1 through 5 are supported, remember there are N^3 quadrature points per element)
		//the constructor then looks like
		//
		//	Kernel_t kernel(d0,d1,d2, bilinear0, bilinear1,..., linear0, linear1, ...)
		//
		//where d0,d1,d2 is the diagonal measurement of the mesh. TODO::probably remove this because the forms have a reference to the mesh?

		//define aliases for linear and bilinear forms.
		//if the form is weighted, define a method with the signature
		//
		//	template<uint64_t N>
		//	constexpr void eval_w(std::array<double,N>& val, const std::array<double,N>& x, const std::array<double,N>& y, const std::array<double,N>& z) const
		//
		//each of the forms use CRTP to avoid virtual dispatch when evaluating the weight. Pass the derived type to DERIVED.
		//if the form is unweighted, pass DERIVED=void. For linear forms, this defaults to a weight of 0 and for bilinear forms (homogeneous rhs), it
		//defaults to a weight of 1 (i.e., unweighted).

		//alias for a linear L2 form
		template<typename DOF_TYPE, typename DERIVED=void>
		using LinearL2_T = LinearL2<Mesh_t, DOF_TYPE, DERIVED>;

		//alias for a symmetric L2 bilinear form
		template<typename DOF_TYPE, typename DERIVED=void>
		using SymmetricL2_T = SymmetricL2<Mesh_t, DOF_TYPE, DERIVED>;

		//alias for a symmetric H1 bilinear form. note that the weight is diagonal for now. TODO: make weight symmetric 3x3 matrix
		template<typename DOF_TYPE, typename DERIVED=void>
		using SymmetricH1_T = SymmetricH1<Mesh_t, DOF_TYPE, DERIVED>;

		//sparse matrices are assembled as CSR in Eigen's format
		//this is done from an intermediate COO_CSR hybrid format so that 
		//building matrices for other libraries shouldn't be too hard
		using SpMat_t = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
		using Vec_t   = Eigen::VectorXd;
	};


}