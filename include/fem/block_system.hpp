#pragma once

#include "fem/forms/bilinear/matrix_multiply.hpp"
#include "util/concepts.hpp"


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
	//create a few shortcuts for blocks
	enum class BlockType {
		Zero,
		Identity,
		ScaledIdentity,
		BilinearForm
	};

	//Primary block type, to be specialized
	template<BlockType type, typename TestHandler_t=void, typename TrialHandler_t=void, typename EvalPolicy=void>
	struct Block;

	//concept for blocks
	template<typename T>
	concept BLOCK_T = std::same_as<std::remove_cvref_t<decltype(T::type)>, BlockType>;

	//Zero block
	template<>
	struct Block<BlockType::Zero>
	{
		uint64_t n_rows, n_cols;
		static constexpr BlockType type = BlockType::Zero;

		template<typename TestHandler_t, typename TrialHandler_t>
		Block(const TestHandler_t& test, const TrialHandler_t& trial) 
			: n_rows(test.n_dofs()), n_cols(trial.n_dofs()) {}

		template<typename TestHandler_t>
		void set_test_handler(const TestHandler_t& test) {n_rows=test.n_dofs();}

		template<typename TrialHandler_t>
		void set_trial_handler(const TrialHandler_t& trial) {n_cols=trial.n_dofs();}
	};

	template<>
	struct Block<BlockType::Identity>
	{
		uint64_t n_rows, n_cols;
		static constexpr BlockType type = BlockType::Identity;

		template<typename TestHandler_t, typename TrialHandler_t>
		Block(const TestHandler_t& test, const TrialHandler_t& trial) 
			: n_rows(test.n_dofs()), n_cols(trial.n_dofs()) {assert(n_rows==n_cols);}

		template<typename TestHandler_t>
		void set_test_handler(const TestHandler_t& test) {n_rows=test.n_dofs();}

		template<typename TrialHandler_t>
		void set_trial_handler(const TrialHandler_t& trial) {n_cols=trial.n_dofs();}
	};

	template<>
	struct Block<BlockType::ScaledIdentity>
	{
		uint64_t n_rows, n_cols;
		static constexpr BlockType type = BlockType::ScaledIdentity;
		double scale{1};

		template<typename TestHandler_t, typename TrialHandler_t>
		Block(const TestHandler_t& test, const TrialHandler_t& trial) 
			: n_rows(test.n_dofs()), n_cols(trial.n_dofs()) {assert(n_rows==n_cols);}

		template<typename TestHandler_t>
		void set_test_handler(const TestHandler_t& test) {n_rows=test.n_dofs();}

		template<typename TrialHandler_t>
		void set_trial_handler(const TrialHandler_t& trial) {n_cols=trial.n_dofs();}
	};

	template<typename TestHandler_type, typename TrialHandler_type, typename EvalPolicy_type>
	struct Block<BlockType::BilinearForm, TestHandler_type, TrialHandler_type, EvalPolicy_type>
	{
		static_assert(!std::is_same_v<TestHandler_type,void>, "BilinearForm Blocks need a test dof handler");
		static_assert(!std::is_same_v<TrialHandler_type,void>, "BilinearForm Blocks need a trial dof handler");
		static_assert(!std::is_same_v<EvalPolicy_type,void>, "BilinearForm Blocks need an evaluation policy");

		using TestHandler_t  = TestHandler_type;
		using TrialHandler_t = TrialHandler_type;
		using EvalPolicy_t   = EvalPolicy_type;

		uint64_t n_rows, n_cols;
		static constexpr BlockType type = BlockType::BilinearForm;

		//need access to the handlers and evaluation policy to construct the requested form
		TestHandler_t const* test_handler{nullptr};
		TrialHandler_t const* trial_handler{nullptr};
		EvalPolicy_t eval_policy{};

		Block(const TestHandler_t& test, const TrialHandler_t& trial) 
			: n_rows(test.n_dofs()), n_cols(trial.n_dofs()), 
				test_handler(&test) , trial_handler(&trial) {}

		void set_test_handler(const TestHandler_t& test) {
			n_rows = test.n_dofs();
			test_handler = &test;
		}

		void set_trial_handler(const TrialHandler_t& trial) {
			n_cols = trial.n_dofs();
			trial_handler = &trial;
		}

		void set_eval_policy(EvalPolicy_t eval) {eval_policy = eval;}
		EvalPolicy_t& get_eval_policy() {return eval_policy;}

		auto build_matvec_form() {
			assert(test_handler!=nullptr); assert(trial_handler!=nullptr);

			using Form_t = BilinearFormMultiply<TestHandler_t,TrialHandler_t,EvalPolicy_t>;
			return Form_t{*test_handler,*trial_handler};
		}
	};




	//container to hold the block system
	//set the system at compile time in dense row-major format
	template<int NROWS, int NCOLS, typename... Blocks>
	struct BlockSystem
	{
		static_assert(sizeof...(Blocks) == NROWS*NCOLS);
		std::tuple<Blocks...> blocks;

		template<int IDX>
		using Block_t = std::tuple_element_t<IDX, std::tuple<Blocks...>>;

		static constexpr int flat_idx(int I, int J) {return J + NCOLS*I;}

		template<int I, int J, typename... Args>
		void set_block(Args&&... args) {
			std::get<flat_idx(I,J)>(blocks) = Block_t<flat_idx(I,J)>(std::forward<Args>(args)...);
		}

		template<int I, int J>
		auto& get_block() {return std::get<flat_idx(I,J)>(blocks);}
		template<int I, int J>
		const auto& get_block() const {return std::get<flat_idx(I,J)>(blocks);}

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
		}

		//get the number of rows before this block
		template<int BLOCK_ROW>
		uint64_t row_offset() const {
			uint64_t N = 0;
			[&]<size_t... Is>(std::index_sequence<Is...>) {
				(N+=get_block<Is,0>().n_rows, ...);
			}(std::make_index_sequence<BLOCK_ROW>{});
			return N;
		}

		//get the number of columns before this block
		template<int BLOCK_COL>
		uint64_t col_offset() const {
			uint64_t M = 0;
			[&]<size_t... Js>(std::index_sequence<Js...>) {
				(M+=get_block<0,Js>().n_cols, ...);
			}(std::make_index_sequence<BLOCK_COL>{});
			return M;
		}

		inline uint64_t total_rows() const {return row_offset<NROWS>();}
		inline uint64_t total_cols() const {return col_offset<NCOLS>();}

		template<int IDX>
		void trivial_block_multiply_accumulate(std::span<double> y, std::span<const double> x) const {
			constexpr BlockType type = Block_t<IDX>::type;
			if constexpr (type==BlockType::Zero || type==BlockType::BilinearForm) {return;}

			constexpr int BLOCK_I = IDX/NCOLS;
			constexpr int BLOCK_J = IDX%NCOLS;
			auto& block = std::get<IDX>(blocks);
			const auto row_start = row_offset<BLOCK_I>();
			const auto col_start = col_offset<BLOCK_J>();
			const auto n_rows = block.n_rows;
			const auto n_cols = block.n_cols;

			std::span<double> y_view = y.subspan(row_start, n_rows);
			std::span<const double> x_view = x.subspan(col_start, n_cols);

			//process this block
			if constexpr (type==BlockType::Identity) {
				assert(n_rows==n_cols);
				#pragma omp simd
				for (size_t idx=0; idx<n_cols; ++idx) {
					y_view[idx] += x_view[idx];
				}
			}
			else if constexpr (type==BlockType::ScaledIdentity) {
				assert(n_rows==n_cols);
				#pragma omp simd
				for (size_t idx=0; idx<n_cols; ++idx) {
					y_view[idx] += block.scale * x_view[idx];
				}
			}
			else {assert(false && "unknown block type");}
		}

		void multiply_accumulate(std::span<double> y, std::span<const double> x) const {
			assert(y.size() == total_rows());
			assert(x.size() == total_cols());

			//handle any trivial blocks
			[&]<size_t... IDXs>(std::index_sequence<IDXs...>) {(
				trivial_block_multiply_accumulate<IDXs>(y,x),...);
			}(std::make_index_sequence<NROWS*NCOLS>{});

			//build the forms and kernel
			auto forms  = [&]<size_t... IDXs>(std::index_sequence<IDXs...>) {
				return std::tuple_cat(make_matvec_form<IDXs>(y,x)...);}(std::make_index_sequence<NROWS*NCOLS>{});
			
			//if there are no bilinear forms, we can return
			if constexpr (std::tuple_size_v<decltype(forms)> == 0) {return;}

			//build the kernel, the forms are already linked to storage
			auto kernel = std::apply([](auto&... form) {
				return Kernel<4>{form...};
			}, forms);

			//set up the loop over elements
			const auto& mesh = kernel.get_mesh();

			auto action = [&](const auto el) {
				kernel.set_element(el);
				kernel.set_basis(el);
				kernel.dispatch_all();
			};

			//TODO: change to parallel over colors
			mesh.for_each_active_element(action);
		};


		template<int IDX>
		auto make_matvec_form(std::span<double> y, std::span<const double> x) {
			//build the matrix-free multiply bilinear form
			constexpr BlockType type = Block_t<IDX>::type;

			//the forms will be concatinated, empty tuples will be discarded
			if constexpr (type!=BlockType::BilinearForm) {return std::tuple<>{};}

			constexpr int BLOCK_I = IDX/NCOLS;
			constexpr int BLOCK_J = IDX%NCOLS;
			auto& block = std::get<IDX>(blocks);
			const auto row_start = row_offset<BLOCK_I>();
			const auto col_start = col_offset<BLOCK_J>();
			const auto n_rows = block.n_rows;
			const auto n_cols = block.n_cols;

			auto form = block.make_matvec_form();
			form.set_global(y.subspan(row_start,n_rows), x.subspan(col_start,n_cols));

			return std::tuple{std::move(form)};
		}
		
	};


	
}