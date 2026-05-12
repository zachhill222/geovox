#pragma once


#include "mesh/voxel_mesh.hpp"
#include "fem/handlers/dofhandler_charms.hpp"
#include "fem/handlers/bc_handler.hpp"
#include "fem/forms/bilinear_H1.hpp"
#include "fem/forms/linear_L2.hpp"
#include "fem/numerics/kernel.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"

#include "util/log_time.hpp"
#include "util/concepts.hpp"
#include "util/point.hpp"

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
		using Point_t     = Point<3,double>;

		using DOF_t       = VoxelQ1<typename Vert_t::PeriodicVariant<BC>>;
		using Handler_t   = DofHandlerCharms<Mesh_t,DOF_t>;
		using BCHandler_t = BCHandler<DOF_t>; 

		using StiffForm   = SymmetricH1<Handler_t>;

		#pragma omp declare simd
		static double rhs_fun(double x, double y, double z) {
			return -100.0*(x*x+y*y+z*z-0.25);}

		#pragma omp declare simd
		static bool rhs_spt(double x, double y, double z) {return (x*x + y*y + z*z < 0.25);}

		using RHSForm = LinearL2<Handler_t,ScatterAction,decltype(&rhs_fun),decltype(&rhs_spt)>;

		using Kernel_t = Kernel<4,TypeList<StiffForm>, TypeList<RHSForm>>;

		using SpMat_t = Eigen::SparseMatrix<double,Eigen::RowMajor>;
		using Vec_t   = Eigen::VectorXd;

		Mesh_t			mesh;
		Handler_t   	dofhandler;
		BCHandler_t     bchandler;
		StiffForm   	stiff_form;
		RHSForm     	rhs_form;
		
		typename StiffForm::MatStorage_t stiff_mat_coo;
		typename RHSForm::VecStorage_t rhs_storage;

		SpMat_t A;
		Vec_t	solution, rhs;

		Poisson(const Point_t low, const Point_t high) :
			mesh{low, high},
			dofhandler{mesh},
			bchandler{},
			stiff_form{dofhandler},
			rhs_form{dofhandler,&rhs_fun, &rhs_spt} {
				stiff_form.set_storage(stiff_mat_coo);
				rhs_form.set_storage(rhs_storage);
			}

		//initialize/reset problem to the specified depth of the mesh
		//the mesh will be in a conformal state after this
		void set_depth(int dd) {
			LogTime timer{"PoissonQ1::set_depth"};
			mesh.set_depth(dd);
			dofhandler.set_depth(dd);
			dofhandler.compress_dof_numbers();
			assert(dofhandler.curr_compressed_dofs().size() == dofhandler.count_dofs() );
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
			LogTime timer{"PoissonQ1::integrate"};
			const auto diag = mesh.high - mesh.low;
			Kernel_t kernel(stiff_form, rhs_form);
			auto action = [this, &kernel](Elem_t el) {
				const auto el_basis = dofhandler.basis_active(el);
				kernel.set_element(el);
				kernel.template B_set_basis<0>(el_basis,el_basis);
				kernel.template B_compute_scatter<0>();

				kernel.template L_set_basis<0>(el_basis);
				kernel.template L_compute_scatter<0>();
			};

			auto predicate = [this](Elem_t el) {return mesh.is_active(el);};
			mesh.template for_each<Elem_t>(action, false, predicate);
		}

		void build_matrices() {
			LogTime timer{"PoissonQ1::build_matrices"};
			const auto& dofs = dofhandler.curr_compressed_dofs();

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
					A = stiff_form.to_eigen_csr(dofs,dofs);
				}

				#ifdef _OPENMP
				#pragma omp task
				#endif
				{
					rhs = rhs_form.to_eigen_Xd(dofs);
				}
			}
			#ifdef _OPENMP
			omp_set_max_active_levels(1);
			omp_set_nested(0);
			#endif
		}

		//cache the boundary dofs
		void cache_bc() {
			LogTime timer{"PoissonQ1::cache_bc"};
			bchandler.cache(dofhandler.curr_compressed_dofs());
		}

		//apply BC to A and the rhs
		void apply_dirichlet() {
			LogTime timer{"PoissonQ1::apply_dirichlet"};
			bchandler.apply(A,rhs,dofhandler.curr_compressed_dofs());
		}

		void solve() {
			LogTime timer{"PoissonQ1::solve"};

			Eigen::ConjugateGradient<SpMat_t, Eigen::Lower|Eigen::Upper> cg;
			cg.compute(A);
			solution = cg.solve(rhs);
			// Eigen::SparseLU<SpMat_t> lu;
			// lu.compute(A);
			// solution = lu.solve(rhs);
		}

		void save_as(const std::string filename) const {
			LogTime timer{"PoissonQ1::save_as"};

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

