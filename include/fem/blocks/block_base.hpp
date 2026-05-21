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
	template<BlockType type, typename TestHandler_type=void, typename TrialHandler_type=void, typename EvalPolicy=void>
	struct Block;

	//concepts for blocks
	template<typename T>
	concept BLOCK_T = std::same_as<std::remove_cvref_t<decltype(T::type)>, BlockType>;


	//Zero block
	template<typename TestHandler_type, typename TrialHandler_type>
	struct Block<BlockType::Zero, TestHandler_type, TrialHandler_type>
	{
		using TestHandler_t  = TestHandler_type;
		using TrialHandler_t = TrialHandler_type;

		uint64_t n_rows{0}, n_cols{0};
		static constexpr BlockType type = BlockType::Zero;

		Block() = default;
		Block(const TestHandler_t& test, const TrialHandler_t& trial) 
			: n_rows(test.n_dofs()), n_cols(trial.n_dofs()) {}

		void set_test_handler(const TestHandler_t& test) {n_rows=test.n_dofs();}
		void set_trial_handler(const TrialHandler_t& trial) {n_cols=trial.n_dofs();}
	};

	template<typename TestHandler_type, typename TrialHandler_type>
	struct Block<BlockType::Identity, TestHandler_type, TrialHandler_type>
	{
		using TestHandler_t  = TestHandler_type;
		using TrialHandler_t = TrialHandler_type;

		uint64_t n_rows{0}, n_cols{0};
		static constexpr BlockType type = BlockType::Identity;

		Block() = default;
		Block(const TestHandler_t& test, const TrialHandler_t& trial) 
			: n_rows(test.n_dofs()), n_cols(trial.n_dofs()) {assert(n_rows==n_cols);}

		void set_test_handler(const TestHandler_t& test) {n_rows=test.n_dofs();}
		void set_trial_handler(const TrialHandler_t& trial) {n_cols=trial.n_dofs();}
	};

	template<typename TestHandler_type, typename TrialHandler_type>
	struct Block<BlockType::ScaledIdentity, TestHandler_type, TrialHandler_type>
	{
		using TestHandler_t  = TestHandler_type;
		using TrialHandler_t = TrialHandler_type;

		uint64_t n_rows{0}, n_cols{0};
		static constexpr BlockType type = BlockType::ScaledIdentity;
		double scale{1};

		Block() = default;
		Block(const TestHandler_t& test, const TrialHandler_t& trial) 
			: n_rows(test.n_dofs()), n_cols(trial.n_dofs()) {assert(n_rows==n_cols);}

		void set_test_handler(const TestHandler_t& test) {n_rows=test.n_dofs();}
		void set_trial_handler(const TrialHandler_t& trial) {n_cols=trial.n_dofs();}
	};

	template<typename TestHandler_type, typename TrialHandler_type, typename EvalPolicy_type>
	struct Block<BlockType::BilinearForm, TestHandler_type, TrialHandler_type, EvalPolicy_type>
	{
		static_assert(!std::is_same_v<TestHandler_type,void>,  "BilinearForm Blocks need a test dof handler");
		static_assert(!std::is_same_v<TrialHandler_type,void>, "BilinearForm Blocks need a trial dof handler");
		static_assert(!std::is_same_v<EvalPolicy_type,void>,   "BilinearForm Blocks need an evaluation policy");

		using TestHandler_t  = TestHandler_type;
		using TrialHandler_t = TrialHandler_type;
		using EvalPolicy_t   = EvalPolicy_type;

		uint64_t n_rows{0}, n_cols{0};
		static constexpr BlockType type = BlockType::BilinearForm;

		//need access to the handlers and evaluation policy to construct the requested form
		TestHandler_t const* test_handler{nullptr};
		TrialHandler_t const* trial_handler{nullptr};
		EvalPolicy_t eval_policy{};

		Block() = default;
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

		auto make_matvec_form() const {
			assert(test_handler!=nullptr); assert(trial_handler!=nullptr);
			
			using Form_t = BilinearFormMultiply<TestHandler_t,TrialHandler_t,EvalPolicy_t>;
			return Form_t{*test_handler,*trial_handler};
		}
	};	
}