#pragma once

#include "gutil.hpp"

#include "mesh/voxel_mesh.hpp"
#include "fem/handlers/dofhandler_charms.hpp"
#include "fem/handlers/bc_handler.hpp"
#include "fem/forms/bilinear/matrix_assembler.hpp"
#include "fem/forms/bilinear/policy_evaluation.hpp"
#include "fem/forms/linear/vector_assembler.hpp"
#include "fem/forms/linear/policy_evaluation.hpp"
#include "fem/numerics/kernel.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"

#include "util/concepts.hpp"

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>

#include <array>
#include <fstream>
#include <string>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	template<typename Mesh_type, uint64_t BC=1>
	struct Poisson
	{
		using Mesh_t      = VoxelMesh<8>;
		using Elem_t      = typename Mesh_t::VoxelElement;
		using Vert_t      = typename Mesh_t::VoxelVertex;
		using Point_t     = gutil::Point<3,double>;

		using DOF_t       = VoxelQ1<typename Vert_t::PeriodicVariant<BC>>;
		using Handler_t   = DofHandlerCharms<Mesh_t,DOF_t>;
		using BCHandler_t = BCHandler<DOF_t>; 

		using H1Eval      = BilinearH1<DOF_t,DOF_t>;
		using StiffForm   = BilinearFormAssembler<Handler_t,Handler_t,H1Eval>;

		#pragma omp declare simd
		static double rhs_fun(double x, double y, double z) {
			return -10.0*(x*x+y*y+z*z-0.25);}

		#pragma omp declare simd
		static bool rhs_spt(double x, double y, double z) {return (x*x + y*y + z*z < 0.25);}

		using RHSEval  = LinearL2<DOF_t,decltype(rhs_fun),decltype(rhs_spt)>;
		using RHSForm  = LinearFormAssembler<Handler_t,RHSEval>;

		using Kernel_t = Kernel<4, StiffForm, RHSForm>;

		using SpMat_t = Eigen::SparseMatrix<double,Eigen::RowMajor>;
		using Vec_t   = Eigen::VectorXd;

		Mesh_t			mesh;
		Handler_t   	dofhandler;
		BCHandler_t     bchandler;
		
		typename StiffForm::MatStorage_t stiff_mat_coo;
		SpMat_t A;
		Vec_t	solution, rhs;

		Poisson(const Point_t low, const Point_t high) :
			mesh{low, high},
			dofhandler{mesh} {}

		//initialize/reset problem to the specified depth of the mesh
		//the mesh will be in a conformal state after this
		void set_depth(int dd) {
			gutil::LogTime timer{"PoissonQ1::set_depth"};
			mesh.set_depth(dd);
			dofhandler.set_depth(dd);
			dofhandler.compress_dof_numbers();
			assert(dofhandler.curr_compressed_dofs().size() == dofhandler.count_dofs() );
			cache_bc();
		}

		//refine the mesh and prolong/interpolate the current solution
		template<typename DOF_Predicate_t = std::nullptr_t>
		void refine(DOF_Predicate_t&& pred = nullptr) {
			for (size_t i=0; i<dofhandler.curr_compressed_dofs().size(); ++i) {
				const DOF_t dof = dofhandler.get_dof(i);
				if constexpr (!NULLPTR_T<DOF_Predicate_t>) {
					if (!pred(dof)) {continue;}
				}

				dofhandler.refine(dof);
			}

			//update mesh and compress dofs
			mesh.process_request_active();
			dofhandler.compress_dof_numbers();

			//transfer solution to the fine grid
			const Vec_t solution_copy = solution;
			solution = Vec_t::Zero(dofhandler.n_dofs());
			dofhandler.update_coefs(solution,solution_copy);
		}

		void integrate() {
			gutil::LogTime timer{"PoissonQ1::integrate"};
			StiffForm stiff_form(dofhandler, dofhandler);
			stiff_mat_coo.clear();
			stiff_form.set_global(stiff_mat_coo);

			RHSForm rhs_form(dofhandler, RHSEval{rhs_fun,rhs_spt});
			rhs = Eigen::VectorXd::Zero(dofhandler.n_dofs());
			rhs_form.set_global(rhs);

			Kernel_t kernel(stiff_form, rhs_form);
			auto action = [&](Elem_t el) {
				const auto el_basis = dofhandler.basis_active(el);
				kernel.set_element(el);
				stiff_form.set_basis(el_basis,el_basis);
				rhs_form.set_basis(el_basis);
				kernel.dispatch_all();
			};

			mesh.template for_each_active_element(action);

			double val=0;
			for (double v : rhs) {val+=v;}
			std::cout << "sum rhs= " << val << "\n";
		}

		void build_matrices() {
			gutil::LogTime timer{"PoissonQ1::build_matrices"};
			const auto& dofs = dofhandler.curr_compressed_dofs();
			A = stiff_mat_coo.to_eigen_csr(dofs,dofs);
		}

		//cache the boundary dofs
		void cache_bc() {
			gutil::LogTime timer{"PoissonQ1::cache_bc"};
			bchandler.cache(dofhandler.curr_compressed_dofs());
		}

		//apply BC to A and the rhs
		void apply_dirichlet() {
			gutil::LogTime timer{"PoissonQ1::apply_dirichlet"};
			bchandler.apply(A,rhs,dofhandler.curr_compressed_dofs());
		}

		void solve() {
			gutil::LogTime timer{"PoissonQ1::solve"};

			Eigen::ConjugateGradient<SpMat_t, Eigen::Lower|Eigen::Upper> cg;
			cg.compute(A);
			solution = cg.solve(rhs);
			// Eigen::SparseLU<SpMat_t> lu;
			// lu.compute(A);
			// solution = lu.solve(rhs);
		}

		void save_as(const std::string filename) const {
			gutil::LogTime timer{"PoissonQ1::save_as"};

			std::ofstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error("PoissonQ1::save_as - could not open file: " + filename);
			}

			//write the mesh and get the number of vertices
			const auto n_verts = mesh.write_unstructured_vtk(file);

			//interpolate the solution to the vertex values
			auto vert_vals = dofhandler.interpolate_to_vertices(solution, n_verts);

			//append solution header
			file << "POINT_DATA " << n_verts << "\n";
			mesh.append_unstructured_point_data_vtk(
				file,
				"SCALARS solution float 1\nLOOKUP_TABLE default",
				n_verts,
				[&vert_vals](Vert_t vtx) {return vert_vals[vtx.linear_index()];});
		}
	};



}

