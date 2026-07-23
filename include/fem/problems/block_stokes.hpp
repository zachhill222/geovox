#pragma once

#include "gutil.hpp"

#include "fem/handlers/dofhandler_charms.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"

#include "fem/blocks/block_base.hpp"
#include "fem/blocks/block_system.hpp"
#include "fem/blocks/block_system_eigen.hpp"

#include "fem/forms/bilinear/policy_evaluation.hpp"
#include "fem/forms/linear/vector_assembler.hpp"
#include "fem/forms/linear/policy_evaluation.hpp"

#include "mesh/voxel_mesh_unstructured.hpp"
#include "mesh/vtk_file_io.hpp"

#include "util/concepts.hpp"

#include <span>
#include <vector>
#include <string>
#include <fstream>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace GV
{
	template<uint64_t V_BC=7, uint64_t MAX_DEPTH=8>
	class BlockStokes
	{
		public:
		using Mesh_t	 	= UnstructuredVoxelMesh<MAX_DEPTH>;
		using Elem_t        = typename Mesh_t::VoxelElement;
		using V_DOF_t    	= VoxelQ1<typename Mesh_t::VoxelVertex::PeriodicVariant<V_BC>>;
		using P_DOF_t    	= VoxelQ1<typename Mesh_t::VoxelVertex::NonPeriodicVariant>;
		using V_Handler_t 	= DofHandlerCharms<Mesh_t,V_DOF_t>;
		using P_Handler_t 	= DofHandlerCharms<Mesh_t,P_DOF_t>;

		using EvalL2        = BilinearL2<P_DOF_t,P_DOF_t>;
		using EvalH1        = BilinearH1<V_DOF_t, V_DOF_t>;
		template<int k>
		using EvalHdiv      = BilinearHdiv<V_DOF_t,P_DOF_t,k>;
		template<int k>
		using EvalHdivAdj   = BilinearHdivAdjoint<P_DOF_t,V_DOF_t,k>;

		//body force terms
		using LinL2         = LinearFormAssembler<V_Handler_t, LinearL2<V_DOF_t>>;
		double fu{0}, fv{0}, fw{0};

		//block system
		using BlockSystem_t = BlockSystem<4,4,
			Block<BlockType::BilinearForm, V_Handler_t, V_Handler_t, EvalH1>,			//(0,0) x-velocity stiffness matrix
			Block<BlockType::Zero, V_Handler_t, V_Handler_t>,							//(0,1)
			Block<BlockType::Zero, V_Handler_t, V_Handler_t>,							//(0,2)
			Block<BlockType::BilinearForm, V_Handler_t, P_Handler_t, EvalHdiv<0>>,		//(0,3) x-pressure gradient matrix

			Block<BlockType::Zero, V_Handler_t, V_Handler_t>,							//(1,0)
			Block<BlockType::BilinearForm, V_Handler_t, V_Handler_t, EvalH1>,			//(1,1) y-velocity stiffness matrix
			Block<BlockType::Zero, V_Handler_t, V_Handler_t>,							//(1,2)
			Block<BlockType::BilinearForm, V_Handler_t, P_Handler_t, EvalHdiv<1>>,		//(1,3) y-pressure gradient matrix

			Block<BlockType::Zero, V_Handler_t, V_Handler_t>,							//(2,0)
			Block<BlockType::Zero, V_Handler_t, V_Handler_t>,							//(2,1)
			Block<BlockType::BilinearForm, V_Handler_t, V_Handler_t, EvalH1>,			//(2,2) z-velocity stiffness matrix
			Block<BlockType::BilinearForm, V_Handler_t, P_Handler_t, EvalHdiv<2>>,		//(2,3) z-pressure gradient matrix

			Block<BlockType::BilinearForm, P_Handler_t, V_Handler_t, EvalHdivAdj<0>>,	//(3,0) x-velocity divergence matrix
			Block<BlockType::BilinearForm, P_Handler_t, V_Handler_t, EvalHdivAdj<1>>,	//(3,1) y-velocity divergence matrix
			Block<BlockType::BilinearForm, P_Handler_t, V_Handler_t, EvalHdivAdj<2>>,	//(3,2) z-velocity divergence matrix
			Block<BlockType::Zero, P_Handler_t, P_Handler_t>>;							//(3,3)


		//define primary problem components
		Mesh_t 			mesh;
		V_Handler_t 	velocity_handler; //all velocity dofs are the same
		P_Handler_t		pressure_handler; 
		BlockSystem_t   system;

		//set solution storage
		std::vector<double> X; //U and P combined

		//constructor
		BlockStokes(const gutil::Point<3,double> low = {0,0,0}, const gutil::Point<3,double> high = {1,1,1}) 
			: mesh{low, high}, velocity_handler{mesh}, pressure_handler{mesh}, system{}
		{
			system.set_test_handlers(velocity_handler, velocity_handler, velocity_handler, pressure_handler);
			system.set_trial_handlers(velocity_handler, velocity_handler, velocity_handler, pressure_handler);

			//apply a trivial pressure BC
			system.template add_bc<3>([](P_DOF_t dof) {return dof.key.depth_linear_index() == 0;}, [](P_DOF_t dof) {return 0.0;});
		}

		//add boundary conditions for velocity
		template<typename Predicate, typename Function = std::nullptr_t>
		void add_velocity_bc(Predicate&& pred, Function&& fun = nullptr) {
			if constexpr (NULLPTR_T<Function>) {
				system.template add_bc<0>(pred, [](V_DOF_t dof) {return 0.0;});
				system.template add_bc<1>(pred, [](V_DOF_t dof) {return 0.0;});
				system.template add_bc<2>(pred, [](V_DOF_t dof) {return 0.0;});
			}
			else {
				system.template add_bc<0>(pred, [fun](V_DOF_t dof) {return fun(dof)[0];});
				system.template add_bc<1>(pred, [fun](V_DOF_t dof) {return fun(dof)[1];});
				system.template add_bc<2>(pred, [fun](V_DOF_t dof) {return fun(dof)[2];});
			}
		}

		//cache the dofs that need the dirichlet bc
		inline void cache_bc() {system.cache_bc();}

		//set the body force
		void set_body_force(const double x, const double y, const double z) {
			fu = x;
			fv = y;
			fw = z;
		}

		//get number of dofs
		size_t n_vel_total() 	  const {return 3*velocity_handler.n_dofs();}
		size_t n_vel_individual() const {return velocity_handler.n_dofs();}
		size_t n_pres() 		  const {return pressure_handler.n_dofs();}
		size_t n_dofs_total() 	  const {return n_vel_total()+n_pres();}

		//access individual components of the solution
		std::span<const double> u() const {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,0,N1);
		}

		std::span<const double> v() const {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,N1,N1);
		}

		std::span<const double> w() const {
			const auto N1 = velocity_handler.n_dofs();
			#ifndef NDEBUG
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			#endif
			return as_span(X,2*N1,N1);
		}

		std::span<const double> p() const {
			const auto N1 = velocity_handler.n_dofs();
			const auto N2 = pressure_handler.n_dofs();
			assert(3*N1+N2 == X.size());
			return as_span(X,3*N1,N2);
		}

		//compute the RHS
		void compute_RHS(std::span<double> rhs) const {
			const auto N = n_vel_individual();
			assert(rhs.size()==n_dofs_total());
			
			LinL2 F_u{velocity_handler, fu};
			LinL2 F_v{velocity_handler, fv};
			LinL2 F_w{velocity_handler, fw};

			F_u.set_global(rhs.subspan(0,N));
			F_v.set_global(rhs.subspan(N,N));
			F_w.set_global(rhs.subspan(2*N,N));

			using Kernel_type = Kernel<4,LinL2,LinL2,LinL2>;
			Kernel_type kernel(F_u,F_v,F_w);

			auto action = [&,this](Elem_t el) {
				kernel.set_element(el);
				const auto v_dofs = velocity_handler.basis_active(el);
				F_u.set_basis(v_dofs);
				F_v.set_basis(v_dofs);
				F_w.set_basis(v_dofs);
				kernel.dispatch_all();
			};

			//perform the loop
			//TODO: parallel by element color?
			// mesh.for_each_active_element(action);
			mesh.for_each_active_element(action);
			
			//set the BC
			system.apply_bc(rhs);
		}


		//initialize the problem to a certain mesh depth
		void set_depth(const uint64_t depth) {
			mesh.set_depth(depth);
			pressure_handler.set_depth(depth);
			mesh.set_depth(depth+1);
			velocity_handler.set_depth(depth+1);

			const auto np = pressure_handler.n_dofs();
			const auto nv = velocity_handler.n_dofs();
			X.assign(3*nv+np,0.0);

			mesh.color_by_index();
			mesh.sort_by_color();

			//ensure the system knows the block sizes and that the pointers to the handlers are still correct
			system.set_test_handlers(velocity_handler, velocity_handler, velocity_handler, pressure_handler);
			system.set_trial_handlers(velocity_handler, velocity_handler, velocity_handler, pressure_handler);
			system.cache_bc();
		}

		//solve the system with no pre-conditioning
		void solve_gmres(int n_iter=50) {
			gutil::LogTime timer{"BlockStokes::solve_gmres"};
			Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_dofs_total());
			compute_RHS(as_span(rhs));

			//set up solver
			using Preconditioner = Eigen::IdentityPreconditioner;
			BlockSystemOperator op(system);
			Eigen::GMRES<decltype(op), Preconditioner> solver;
			solver.setMaxIterations(n_iter);
			solver.compute(op);

			Eigen::Map<Eigen::VectorXd> X_map(X.data(), X.size());
			X_map = solver.solve(rhs).eval();
		}

		//save to file
		void save_as(const std::string filename) const {
			gutil::LogTime timer{"BlockStokes::save_as"};

			//save mesh topology
			mesh.collect_vertices();
			mesh.save_as_binary(filename);

			//interpolate solution to vertex values
			auto u_vals = velocity_handler.interpolate_to_vertices(u(), mesh.get_vertices());
			auto v_vals = velocity_handler.interpolate_to_vertices(v(), mesh.get_vertices());
			auto w_vals = velocity_handler.interpolate_to_vertices(w(), mesh.get_vertices());
			auto p_vals = pressure_handler.interpolate_to_vertices(p(), mesh.get_vertices());

			auto v_lookup = make_index_lookup<std::array<double,3>>(
				[&](uint64_t idx){return std::array<double,3>{u_vals[idx], v_vals[idx], w_vals[idx]};}, "velocity");
			auto p_lookup = make_index_lookup<double>([&](uint64_t idx){return p_vals[idx];}, "pressure");

			mesh.append_point_data_field_binary(filename, "solution", v_lookup, p_lookup);
		}
	};
}