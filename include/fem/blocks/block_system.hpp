#pragma once

#include "fem/blocks/block_base.hpp"
#include "fem/numerics/kernel.hpp"
#include "fem/handlers/bc_handler.hpp"
#include "fem/forms/bilinear/matrix_multiply.hpp"

#include "util/concepts.hpp"
#include "util/log_time.hpp"



#include <type_traits>
#include <cstdint>
#include <span>
#include <tuple>
#include <vector>
#include <concepts>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//container to hold the block system
	//set the system at compile time in dense row-major format
	template<size_t NROWS, size_t NCOLS, typename... Blocks>
	struct BlockSystem
	{
		//access type information
		//use the row-major convection that block(i,j) has the flat index k = j + NCOLS*i;
		template<size_t IDX>
		using Block_t = std::tuple_element_t<IDX, std::tuple<Blocks...>>;

		template<size_t BLOCK_ROW>
		using TestHandler_t = typename Block_t<BLOCK_ROW*NCOLS>::TestHandler_t;

		template<size_t BLOCK_COL>
		using TrialHandler_t = typename Block_t<BLOCK_COL>::TrialHandler_t;

		template<size_t BLOCK_ROW>
		using RowDOF_t = typename TestHandler_t<BLOCK_ROW>::DOF_t;

		template<size_t BLOCK_COL>
		using ColDOF_t = typename TrialHandler_t<BLOCK_COL>::DOF_t;

		//default constructor to set nullpointers
		BlockSystem() {
			//set test handlers to nullptr
			[&]<size_t... Rs>(std::index_sequence<Rs...>) {
				((std::get<Rs>(test_dof_handlers) = nullptr),...);
			}(std::make_index_sequence<NROWS>{});

			//set trial handlers to nullptr
			[&]<size_t... Cs>(std::index_sequence<Cs...>) {
				((std::get<Cs>(trial_dof_handlers) = nullptr),...);
			}(std::make_index_sequence<NCOLS>{});
		}


		//store and access blocks
		static_assert(sizeof...(Blocks) == NROWS*NCOLS);
		std::tuple<Blocks...> blocks;

		template<size_t I, size_t J, typename... Args>
		void set_block(Args&&... args) {
			std::get<J+NCOLS*I>(blocks) = Block_t<J+NCOLS*I>(std::forward<Args>(args)...);
		}

		template<size_t I, size_t J>
		auto& get_block() {return std::get<J+NCOLS*I>(blocks);}
		
		template<size_t I, size_t J>
		const auto& get_block() const {return std::get<J+NCOLS*I>(blocks);}

		static constexpr size_t count_bilinear_forms() {
			return ((Blocks::type == BlockType::BilinearForm ? 1 : 0) + ...);
		}

		//store and access pointers to dof handlers
		template<size_t... Rs>
		static auto make_test_handler_tuple(std::index_sequence<Rs...>) -> std::tuple<TestHandler_t<Rs> const*...>;
		decltype(make_test_handler_tuple(std::make_index_sequence<NROWS>{})) test_dof_handlers; 

		template<size_t... Cs>
		static auto make_trial_handler_tuple(std::index_sequence<Cs...>) -> std::tuple<TrialHandler_t<Cs> const*...>;
		decltype(make_trial_handler_tuple(std::make_index_sequence<NCOLS>{})) trial_dof_handlers;

		template<typename... TestHandler_ts>
		void set_test_handlers(const TestHandler_ts&... test_handlers) {
			static_assert(sizeof...(TestHandler_ts)==NROWS);
			//pack references to the handlers into a tuple
			auto h_tuple = std::tie(test_handlers...);
			//for the i-th block row, pass the i-th test handler
			//the loop must be unpacked at compile time
			[&]<size_t... IDXs>(std::index_sequence<IDXs...>) {
				(std::get<IDXs>(blocks).set_test_handler(
					std::get<IDXs/NCOLS>(h_tuple)	//note we are using row-major, IDX/NCOLS is the row number
				), ...);
			}(std::make_index_sequence<NROWS*NCOLS>{});

			//store pointers to the handlers
			[&]<size_t... Rs>(std::index_sequence<Rs...>) {
				((std::get<Rs>(test_dof_handlers) = &std::get<Rs>(h_tuple)), ...);
			}(std::make_index_sequence<NROWS>{});
		}

		template<typename... TrialHandler_ts>
		void set_trial_handlers(const TrialHandler_ts&... trial_handlers) {
			static_assert(sizeof...(TrialHandler_ts)==NCOLS);
			//pack references to the handlers into a tuple
			auto h_tuple = std::tie(trial_handlers...);
			//for the i-th block row, pass the i-th test handler
			//the loop must be unpacked at compile time
			[&]<size_t... IDXs>(std::index_sequence<IDXs...>) {
				(std::get<IDXs>(blocks).set_trial_handler(
					std::get<IDXs%NCOLS>(h_tuple)	//note we are using row-major, IDX%NCOLS is the col number
				), ...);
			}(std::make_index_sequence<NROWS*NCOLS>{});

			//store pointers to the handlers
			[&]<size_t... Cs>(std::index_sequence<Cs...>) {
				((std::get<Cs>(trial_dof_handlers) = &std::get<Cs>(h_tuple)), ...);
			}(std::make_index_sequence<NCOLS>{});
		}


		//set and track essential boundary conditions
		template<size_t... Rs>
		static auto make_bc_tuple(std::index_sequence<Rs...>) -> std::tuple<BCHandler<RowDOF_t<Rs>>...>;
		decltype(make_bc_tuple(std::make_index_sequence<NROWS>{})) bc_handlers;

		template<size_t BLOCK_ROW>
		auto& get_bc_handler() {return std::get<BLOCK_ROW>(bc_handlers);}

		template<size_t BLOCK_ROW>
		const auto& get_bc_handler() const {return std::get<BLOCK_ROW>(bc_handlers);}

		template<size_t BLOCK_ROW, typename... Args>
		void add_bc(Args&&... args) {
			get_bc_handler<BLOCK_ROW>().add_essential(std::forward<Args>(args)...);
		}

		void cache_bc() {
			[&]<size_t... Rs>(std::index_sequence<Rs...>) {
				(get_bc_handler<Rs>().cache(std::get<Rs>(test_dof_handlers) -> curr_compressed_dofs() ), ...);
			}(std::make_index_sequence<NROWS>{});
		}

		void apply_bc_matvec(std::span<double> y, std::span<const double> x, const double alpha=1.0) const {
			//copy the values of x into y at the active boundary condition dofs
			[&]<size_t... Rs>(std::index_sequence<Rs...>) {
				(get_bc_handler<Rs>().apply_matvec(
					y.subspan(row_offset<Rs>(), block_n_rows<Rs>()),
					x.subspan(row_offset<Rs>(), block_n_rows<Rs>()),
					alpha),...);
			}(std::make_index_sequence<NROWS>{});
		}

		void apply_bc(std::span<double> y) const {
			//set the values of y at active boundary dofs to their specified value (in the bc struct)
			[&]<size_t... Rs>(std::index_sequence<Rs...>) {
				(get_bc_handler<Rs>().apply(
					y.subspan(row_offset<Rs>(), block_n_rows<Rs>()),
					std::get<Rs>(test_dof_handlers) -> curr_compressed_dofs()),...);
			}(std::make_index_sequence<NROWS>{});
		}

		//get the number of rows before this block
		template<size_t BLOCK_ROW>
		uint64_t row_offset() const {
			uint64_t N = 0;
			[&]<size_t... Is>(std::index_sequence<Is...>) {
				((N+=get_block<Is,0>().n_rows), ...);
			}(std::make_index_sequence<BLOCK_ROW>{});
			return N;
		}

		template<size_t BLOCK_ROW>
		inline uint64_t block_n_rows() const {
			return get_block<BLOCK_ROW,0>().n_rows;
		}

		//get the number of columns before this block
		template<size_t BLOCK_COL>
		uint64_t col_offset() const {
			uint64_t M = 0;
			[&]<size_t... Js>(std::index_sequence<Js...>) {
				((M+=get_block<0,Js>().n_cols), ...);
			}(std::make_index_sequence<BLOCK_COL>{});
			return M;
		}

		template<size_t BLOCK_COL>
		inline uint64_t block_n_cols() const {
			return get_block<0,BLOCK_COL>().n_cols;
		}

		inline uint64_t total_rows() const {return row_offset<NROWS>();}
		inline uint64_t total_cols() const {return col_offset<NCOLS>();}

		template<size_t IDX>
		void trivial_block_multiply_accumulate(std::span<double> y, std::span<const double> x, const double alpha=1.0) const {
			constexpr BlockType type = Block_t<IDX>::type;
			if constexpr (type==BlockType::Zero || type==BlockType::BilinearForm) {return;}

			constexpr size_t BLOCK_I = IDX/NCOLS;
			constexpr size_t BLOCK_J = IDX%NCOLS;
			const auto& block        = std::get<IDX>(blocks);
			const auto row_start     = row_offset<BLOCK_I>();
			const auto col_start     = col_offset<BLOCK_J>();
			const auto n_rows        = block.n_rows;
			const auto n_cols        = block.n_cols;

			std::span<double> y_view = y.subspan(row_start, n_rows);
			std::span<const double> x_view = x.subspan(col_start, n_cols);

			//process this block
			if constexpr (type==BlockType::Identity) {
				assert(n_rows==n_cols);
				#pragma omp simd
				for (size_t idx=0; idx<n_cols; ++idx) {
					y_view[idx] += alpha * x_view[idx];
				}
			}
			else if constexpr (type==BlockType::ScaledIdentity) {
				assert(n_rows==n_cols);
				const double scale = alpha * block.scale;
				#pragma omp simd
				for (size_t idx=0; idx<n_cols; ++idx) {
					y_view[idx] += scale * x_view[idx];
				}
			}
			else {assert(false && "unknown block type");}
		}

		void multiply_accumulate(std::span<double> y, std::span<const double> x, const double alpha=1.0) const {
			assert(y.size() == total_rows());
			assert(x.size() == total_cols());

			//handle any trivial blocks
			[&]<size_t... IDXs>(std::index_sequence<IDXs...>) {(
				trivial_block_multiply_accumulate<IDXs>(y,x,alpha),...);
			}(std::make_index_sequence<NROWS*NCOLS>{});

			if constexpr (count_bilinear_forms() == 0) {return;}
			const bool use_parallel = total_rows()>1e5;
			
			#pragma omp parallel if (use_parallel)
			{
			//build the forms and kernel
			auto forms  = [&]<size_t... IDXs>(std::index_sequence<IDXs...>) {
				return std::tuple_cat(make_matvec_form<IDXs>(y,x,alpha)...);}(std::make_index_sequence<NROWS*NCOLS>{});

			//build the kernel, the forms are already linked to storage
			auto kernel = std::apply([](auto&... form) {
				return Kernel<4, std::remove_reference_t<decltype(form)>...>{form...};
			}, forms);

			//set up the loop over elements
			const auto& mesh = kernel.get_mesh();

			auto action = [&](const auto el) {
				kernel.set_element(el);
				kernel.set_basis(el);
				kernel.dispatch_all();
			};

			//TODO: change to parallel over colors
			if (use_parallel) {mesh.for_each_active_element_color_omp(action);}
			else {mesh.for_each_active_element(action);}
			}
			//apply any boundary conditions
			apply_bc_matvec(y,x,alpha);
		};


		template<size_t IDX>
		auto make_matvec_form(std::span<double> y, std::span<const double> x, const double scale=1.0) const {
			using tag_type = std::conditional_t<Block_t<IDX>::type == BlockType::BilinearForm, std::true_type, std::false_type>;
			return make_matvec_form_impl<IDX>(y,x,scale,tag_type{});
		}

		template<size_t IDX>
		auto make_matvec_form_impl(std::span<double> y, std::span<const double> x, const double scale=1.0, std::true_type tag=true) const {
			static_assert(Block_t<IDX>::type == BlockType::BilinearForm);

			constexpr size_t BLOCK_I = IDX/NCOLS;
			constexpr size_t BLOCK_J = IDX%NCOLS;
			const auto& block        = std::get<IDX>(blocks);
			const auto row_start     = row_offset<BLOCK_I>();
			const auto col_start     = col_offset<BLOCK_J>();
			const auto n_rows        = block.n_rows;
			const auto n_cols        = block.n_cols;

			auto form = block.make_matvec_form();
			form.set_global(y.subspan(row_start,n_rows), x.subspan(col_start,n_cols));

			return std::tuple{std::move(form)};
		}

		template<size_t IDX>
		auto make_matvec_form_impl(std::span<double> y, std::span<const double> x, const double scale=1.0, std::false_type tag=false) const {
			static_assert(Block_t<IDX>::type != BlockType::BilinearForm);
			return std::tuple<>{};
		}

	};


	
}