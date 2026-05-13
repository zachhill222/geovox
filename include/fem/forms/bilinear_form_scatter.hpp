#pragma once

#include "fem/numerics/csr_storage.hpp"

#include <vector>
#include <span>
#include <cstdint>
#include <cassert>
#include <type_traits>

namespace GV
{	
	//a class responsible for scattering a local matrix result to a global matrix
	//TODO: can we make this more efficient with a prepare(..) method?
	template<typename TestDOF_t, typename TrialDOF_t, typename LocalMatOwner>
	struct BilinearFormScatterLocalMat
	{
		//link to global storage
		using MatStorage_t = CSR_COO<TestDOF_t,TrialDOF_t>;
		using MatRow_t     = typename MatStorage_t::Row_t;
		
		MatStorage_t* global_mat = nullptr;
		inline void set_global(MatStorage_t& coo) {global_mat = &coo;}

		//link to local storage
		const LocalMatOwner& local_mat;

		//constructors with links
		BilinearFormScatterLocalMat(const LocalMatOwner& LM) : local_mat(LM) {}

		//scatter local to global
		//note the COO storage type uses the DOFs directly as indices/keys
		void scatter(std::span<const TestDOF_t> test, std::span<const TrialDOF_t> trial) const {
			const uint64_t n_test  = local_mat.n_test;
			const uint64_t m_trial = local_mat.m_trial;

			assert(global_mat != nullptr);
			assert(test.size()  == n_test);
			assert(trial.size() == m_trial);

			//add the results of the local matrix to the global matrix
			//preserves sorted and accumulated
			for (uint64_t i=0; i<n_test; ++i) {
				MatRow_t& row = global_mat->get_row(test[i]);
				row.reserve(row.size()+m_trial);
				for (uint64_t j=0; j<m_trial; ++j) {
					row.emplace_back(trial[j], local_mat.value(i,j));
				}
				row.accumulate();
			}
		}
	};


	//a class responsible for scattering a local vector result to a global vector
	//generally the handler is for the test dofs.
	template<typename DOF_t, bool ACCUMULATE>
	struct BilinearFormScatterLocalVec
	{
		//link to global storage
		std::span<double> global_vec;
		inline void set_global(std::span<double> vec) {global_vec = vec;}

		//link to local storage
		std::span<const double> local_vec;
		inline void set_local(std::span<const double> vec) {local_vec = vec;}

		//scatter local to global
		void scatter(std::span<const uint64_t> loc2glob) const {
			const uint64_t n_dofs = loc2glob.size();
			assert(local_vec.size() == n_dofs);

			for (uint64_t i=0; i<n_dofs; ++i) {
				const uint64_t I = loc2glob[i];
				assert(I<global_vec.size());
				if constexpr (ACCUMULATE) {global_vec[I] += local_vec[i];}
				else {global_vec[I] = local_vec[i];}
			}
		}
	};
}