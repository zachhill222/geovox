#pragma once

#include <vector>
#include <span>
#include <cstdint>
#include <cassert>
#include <type_traits>

namespace GV
{	
	//a class responsible for scattering a local vector result to a global vector
	//generally the handler is for the test dofs.
	template<typename TestDOF_t>
	struct LinearFormScatterToGlobalVector
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
				global_vec[I] += local_vec[i];
			}
		}
	};
}