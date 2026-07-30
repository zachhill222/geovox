#include "gutil.hpp"

#include "fem/handlers/dofhandler_charms.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"
#include "fem/numerics/kernel.hpp"
#include "mesh/voxel_mesh.hpp"
#include "fem/forms/bilinear/policy_evaluation.hpp"
#include "fem/forms/bilinear/matrix_multiply.hpp"

#include "util/compatibility.hpp"

using Mesh_t    = GV::VoxelMesh<10>;
using Elem_t    = Mesh_t::VoxelElement;
using Vert_t    = Mesh_t::VoxelVertex;
using DofKey_t  = GV::VoxelVertexKey<10,0>;
using DOF_t     = GV::VoxelQ1<DofKey_t>;
using Handler_t = GV::DofHandlerCharms<Mesh_t,DOF_t>;

using H1Eval_t      = GV::BilinearH1<DOF_t,DOF_t>;
using L2Eval_t      = GV::BilinearL2<DOF_t,DOF_t>;
using HdivEval_t    = GV::BilinearHdiv<DOF_t,DOF_t,0>;
using HdivAdjEval_t = GV::BilinearHdivAdjoint<DOF_t,DOF_t,0>;

using BiMass_t      = GV::BilinearFormMultiply<Handler_t, Handler_t, L2Eval_t>;
using BiStiff_t 	= GV::BilinearFormMultiply<Handler_t, Handler_t, H1Eval_t>;
using DivForm_t 	= GV::BilinearFormMultiply<Handler_t, Handler_t, HdivEval_t>;
using DivFormAdj_t 	= GV::BilinearFormMultiply<Handler_t, Handler_t, HdivAdjEval_t>;

using Kernel_t  = GV::Kernel<5,BiMass_t,BiStiff_t,DivForm_t,DivFormAdj_t>;

int main(int argc, char* argv[]) {
	gutil::LogTime t0{"Program"};

	//uniform depth
	const int depth = 4;

	//define mesh
	Mesh_t mesh({0,0,0}, {1,2,3});
	mesh.set_depth(depth);

	//define dofhandler
	Handler_t dofhandler(mesh);
	dofhandler.set_depth(depth);

	//bilinear forms
	BiMass_t     mass_bl(dofhandler,dofhandler);
	BiStiff_t    stiff_bl(dofhandler,dofhandler);
	DivForm_t    div_bl(dofhandler,dofhandler);
	DivFormAdj_t diva_bl(dofhandler,dofhandler);

	//kernel
	Kernel_t kernel(mass_bl, stiff_bl, div_bl, diva_bl);

	//vectors to multiply
	const auto N = dofhandler.n_dofs();
	Eigen::VectorXd ones = Eigen::VectorXd::Ones(N);
	Eigen::VectorXd x(N);
	int i=0;
	auto fill_x = [&x, &i](Vert_t vtx){x[i++] = vtx.x();};
	mesh.template for_each_depth<Vert_t>(depth, fill_x);

	//result vectors
	Eigen::VectorXd M1 = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd Ax = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd A1 = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd B1 = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd Bax= Eigen::VectorXd::Zero(N);

	mass_bl.set_global(M1,ones);
	stiff_bl.set_global(Ax,x);
	stiff_bl.set_global(A1,ones);
	div_bl.set_global(B1,ones);
	diva_bl.set_global(Bax, x);

	//integrate action
	auto action = [&](Elem_t el) {
		const auto dofs = dofhandler.basis_active(el);
		kernel.set_element(el);

		mass_bl.set_basis(dofs,dofs);
		stiff_bl.set_basis(dofs,dofs);
		div_bl.set_basis(dofs,dofs);
		diva_bl.set_basis(dofs,dofs);

		kernel.dispatch_all();
	};

	mesh.template for_each_depth<Elem_t>(depth, action);

	//compute dot products
	std::cout << "1M1= " << ones.dot(M1) << std::endl;
	std::cout << "xAx= " << x.dot(Ax) << std::endl;
	std::cout << "1A1= " << ones.dot(A1) << std::endl;
	std::cout << "xB1= " << x.dot(B1) << std::endl;
	std::cout << "1Bax= "<< ones.dot(Bax) << std::endl;
}