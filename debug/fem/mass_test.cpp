#include "fem/handlers/dofhandler_charms.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"
#include "fem/numerics/kernel.hpp"
#include "fem/numerics/csr_storage.hpp"
#include "mesh/voxel_mesh.hpp"
#include "fem/forms/bilinear/matrix_assembler.hpp"
#include "fem/forms/bilinear/policy_evaluation.hpp"
#include "fem/forms/linear/vector_assembler.hpp"
#include "fem/forms/linear/policy_evaluation.hpp"
#include "util/log_time.hpp"


using Mesh_t    = GV::VoxelMesh<10>;
using Elem_t    = Mesh_t::VoxelElement;
using Vert_t    = Mesh_t::VoxelVertex;
using DofKey_t  = GV::VoxelVertexKey<10,0>;
using DOF_t     = GV::VoxelQ1<DofKey_t>;

using Handler_t = GV::DofHandlerCharms<Mesh_t,DOF_t>;

using H1Eval_t  = GV::BilinearH1<DOF_t,DOF_t>;
using L2Eval_t  = GV::BilinearL2<DOF_t,DOF_t>;
using BiMass_t  = GV::BilinearFormAssembler<Handler_t,Handler_t,L2Eval_t>;
using BiStiff_t = GV::BilinearFormAssembler<Handler_t,Handler_t,H1Eval_t>;

int main(int argc, char* argv[]) {
	GV::LogTime t0{"Program"};

	//uniform depth
	const int depth = 4;

	//define mesh
	Mesh_t mesh({0,0,0}, {1,2,3});
	mesh.set_depth(depth);

	//define dofhandler
	Handler_t dofhandler(mesh);
	dofhandler.set_depth(depth);
	dofhandler.compress_dof_numbers();

	//bilinear forms
	BiMass_t  mass_bl(dofhandler, dofhandler);
	BiStiff_t stif_bl(dofhandler, dofhandler);

	GV::CSR_COO<DOF_t,DOF_t> mass_global_coo, stiff_global_coo;
	mass_bl.set_global(mass_global_coo);
	stif_bl.set_global(stiff_global_coo);

	//kernel
	GV::Kernel<4,BiMass_t,BiStiff_t> kernel(mass_bl, stif_bl);

	//integrate bilinear forms
	auto integrate = [&](Elem_t el) {
		const auto el_basis = dofhandler.basis_active(el);
		
		kernel.set_element(el);
		mass_bl.set_basis(el_basis,el_basis);
		stif_bl.set_basis(el_basis,el_basis);
		kernel.dispatch_all();
	};

	{
		GV::LogTime time{"build COO_CSR"};
		mesh.for_each_depth<Elem_t>(depth,integrate);
	}
	
	Eigen::SparseMatrix<double,Eigen::RowMajor,int> mass_mat, stiff_mat;

	#ifdef _OPENMP
	omp_set_max_active_levels(2);
	omp_set_nested(1);
	#pragma omp parallel
	#pragma omp single
	#endif
	{
		#ifdef _OPENMP
		#pragma omp task
		#endif
		{
			mass_mat = mass_global_coo.to_eigen_csr(dofhandler.curr_compressed_dofs(),dofhandler.curr_compressed_dofs());
		}

		#ifdef _OPENMP
		#pragma omp task
		#endif
		{
			stiff_mat = stiff_global_coo.to_eigen_csr(dofhandler.curr_compressed_dofs(),dofhandler.curr_compressed_dofs());
		}
	}

	Eigen::VectorXd vec = Eigen::VectorXd::Ones(mass_mat.rows());
	std::cout << "mass: " << (mass_mat * vec).transpose() * vec << std::endl;


	//populate the vec with the x coordinates of each dof to test the stiffness matrix
	for (size_t i=0; i<dofhandler.curr_compressed_dofs().size(); ++i) {
		double x = dofhandler.curr_compressed_dofs()[i].key.x();
		vec[i] = (1.0-x)*mesh.low()[0] + x*mesh.high()[0];
	}

	std::cout << "stiff: " << (stiff_mat * vec).transpose() * vec << std::endl;

}