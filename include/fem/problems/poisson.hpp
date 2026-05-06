#pragma once

#include "fem/problems/problem_base.hpp"
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
	template<typename Mesh_type>
	struct Poisson : BaseProblem<Mesh_type>
	{
		using BASE = BaseProblem<Mesh_type>;

		using Mesh_t      = typename BASE::Mesh_t;
		using Elem_t      = typename BASE::Elem_t;
		using Vert_t      = typename BASE::Vert_t;
		using Point_t     = Point<3,double>;

		using DOF_t       = VoxelQ1<Vert_t>;
		using Handler_t   = DofHandlerCharms<Mesh_t,DOF_t>;
		using BCHandler_t = BCHandler<DOF_t>; 

		using StiffForm   = SymmetricH1<Handler_t>;

		struct RHSForm : public LinearL2<Handler_t,ScatterAction,RHSForm>
		{
			using BASE_FORM = LinearL2<Handler_t,ScatterAction,RHSForm>;
			using BASE_FORM::BASE_FORM;
		
			template<uint64_t N>
			constexpr void eval_w(std::array<double,N>& val, const std::array<double,N>& x, const std::array<double,N>& y, const std::array<double,N>& z) const
			{
				#pragma omp simd
				for (uint64_t i=0; i<N; ++i) {
					val[i] = x[i]+y[i]+z[i];
				}
			}
		};

		using Kernel_t = Kernel<4,TypeList<StiffForm>, TypeList<RHSForm>>;

		using SpMat_t = typename BASE::SpMat_t;
		using Vec_t   = typename BASE::Vec_t;

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
			stiff_form{mesh},
			rhs_form{mesh} {
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
			assert(dofhandler.curr_compressed_dofs().size() == dofhandler.n_dofs() );
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

		//apply BC to A and the rhs
		void apply_dirichlet() {
			LogTime timer{"PoissonQ1::apply_dirichlet"};
			bchandler.apply(A,rhs,dofhandler.curr_compressed_dofs());
		}

		void solve() {
			LogTime timer{"PoissonQ1::solve"};

			// Eigen::ConjugateGradient<SpMat_t, Eigen::Lower|Eigen::Upper> cg;
			// cg.compute(A);
			// solution = cg.solve(rhs);
			Eigen::SparseLU<SpMat_t> lu;
			lu.compute(A);
			solution = lu.solve(rhs);
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

