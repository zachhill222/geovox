#include "fem/handlers/dofhandler_charms.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"
#include "fem/numerics/kernel.hpp"
#include "mesh/voxel_mesh.hpp"
#include "fem/forms/bilinear_L2.hpp"
#include "fem/forms/bilinear_H1.hpp"
#include "util/log_time.hpp"


using Mesh_t    = GV::VoxelMesh<10>;
using Elem_t    = Mesh_t::VoxelElement;
using Vert_t    = Mesh_t::VoxelVertex;
using DofKey_t  = GV::VoxelVertexKey<11,0,0>;
using DOF_t     = GV::VoxelQ1<DofKey_t>;
using Handler_t = GV::DofHandlerBase<Mesh_t,DOF_t>;

using BiMass_t  = GV::SymmetricL2<Handler_t>;
using BiStiff_t = GV::SymmetricH1<Handler_t>;

struct MassKernel : public GV::SymmetricL2<Handler_t, GV::ScatterAction, MassKernel>
{
	using BASE = GV::SymmetricL2<Handler_t, GV::ScatterAction, MassKernel>;
	using BASE::BASE;

	template<uint64_t N>
	constexpr void eval_w(
			std::array<double,N>& w_val,
			const std::array<double,N>& x,
			const std::array<double,N>& y,
			const std::array<double,N>& z) const {
		#pragma omp simd
		for (uint64_t i=0; i<N; ++i) {
			//evaluate weight
			w_val[i] = 1.0;
		}
	}
};

using Kernel_t  = GV::Kernel<5,GV::TypeList<MassKernel,BiStiff_t>, GV::TypeList<>>;

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
	MassKernel mass_bl(dofhandler);
	typename MassKernel::MatStorage_t mass_global_coo;
	mass_bl.set_mat(mass_global_coo);

	BiStiff_t stiff_bl(dofhandler);
	typename BiStiff_t::MatStorage_t stiff_global_coo;
	stiff_bl.set_mat(stiff_global_coo);

	//kernel
	const auto diag = mesh.high - mesh.low;
	Kernel_t kernel(diag[0], diag[1], diag[2], mass_bl, stiff_bl);

	//integrate bilinear forms
	auto integrate = [&kernel, &dofhandler](Elem_t el) {
		const auto basis0 = DOF_t::dofs_on_elem(el);
		std::vector<DOF_t> el_basis;
		for (DOF_t b : basis0) {
			if (b.exists() and dofhandler.is_active(b)) {el_basis.push_back(b);}
		}

		kernel.set_element(el);
		kernel.B_form<0>().set_basis(el_basis,el_basis);
		kernel.B_form<1>().set_basis(el_basis,el_basis);
		kernel.B_compute_scatter<0>();
		kernel.B_compute_scatter<1>();
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
			mass_mat = mass_bl.to_eigen_csr(dofhandler.curr_compressed_dofs(),dofhandler.curr_compressed_dofs());
		}

		#ifdef _OPENMP
		#pragma omp task
		#endif
		{
			stiff_mat = stiff_bl.to_eigen_csr(dofhandler.curr_compressed_dofs(),dofhandler.curr_compressed_dofs());
		}
	}

	Eigen::VectorXd vec = Eigen::VectorXd::Ones(mass_mat.rows());
	std::cout << "mass: " << (mass_mat * vec).transpose() * vec << std::endl;


	//populate the vec with the x coordinates of each dof to test the stiffness matrix
	for (size_t i=0; i<dofhandler.curr_compressed_dofs().size(); ++i) {
		double x = dofhandler.curr_compressed_dofs()[i].key.x();
		vec[i] = (1.0-x)*mesh.low[0] + x*mesh.high[0];
	}

	std::cout << "stiff: " << (stiff_mat * vec).transpose() * vec << std::endl;

}