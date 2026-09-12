#pragma once

#include "gutil.hpp"

#include "util/util.hpp"

namespace GV {

	//////////////////////////////////////////////////////////////////
	/// When building a sparse matrix, it is usually most convenient to 
	/// build it from triplets. We make a custom triplet type to help
	/// de-duplicate entries before passing to e.g., Eigen.
	///
	/// Set StorageOrder to 0 for ColMajor and 1 for RowMajor.
	/// The sorting order is outer index then inner index (eg. col then row for ColMajor)
	/// This allows us to use Eigen's setFromSortedTriplets direcly from a vector
	/// of GV::Triplet.
	//////////////////////////////////////////////////////////////////
	template<typename Scalar_t, typename StorageIndex=size_t, int StorageOrder=0> requires (StorageOrder==0 || StorageOrder==1)
	struct Triplet {
		//add an api compatible with Eigen
		[[nodiscard]] StorageIndex row() const noexcept {return i;}
		[[nodiscard]] StorageIndex col() const noexcept {return j;}
		[[nodiscard]] Scalar_t     value() const noexcept {return val;}

		StorageIndex i, j;
		Scalar_t val;

		constexpr Triplet(StorageIndex i, StorageIndex j, Scalar_t v) noexcept : i(i), j(j), val(v) {}
		constexpr Triplet() : i(0), j(0), val(0) {}
		static constexpr Triplet None() noexcept {return Triplet{StorageIndex(-1), StorageIndex(-1), Scalar_t{}};}

		bool operator<(const Triplet& other) const noexcept requires (StorageOrder==0) {
			if (j<other.j) {return true;}
			if (j==other.j && i < other.i) {return true;}
			return false;
		}

		bool operator<(const Triplet& other) const noexcept requires (StorageOrder==1) {
			if (i<other.i) {return true;}
			if (i==other.i && j < other.j) {return true;}
			return false;
		}

		bool operator==(const Triplet& other) const noexcept {
			return i==other.i && j==other.j;
		}

		//////////////////////////////////////////////////////////////////////////////////////////////
		/// Debug and convenience methods
		//////////////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] std::string to_string() const noexcept {
			std::string result = "(";
			result += std::to_string(val) + ", " + std::to_string(i) + ", " + std::to_string(j) + ")";
			return result;
		}

		//////////////////////////////////////////////////////////////////////////////////////////////
		/// Static methods to de-duplicate entries
		//////////////////////////////////////////////////////////////////////////////////////////////
		static void Compress(std::vector<Triplet>& list) {
			//We assume that we are already in a multithreaded region
			if (list.empty()) {return;}

			std::sort(list.begin(), list.end());
			auto accumulate_into = list.begin();

			for (auto it=list.begin()+1; it!=list.end(); ++it) {
				if (*it == *accumulate_into) {
					accumulate_into->val += it->val;
				}
				else {
					//keep the block to keep contiguous at the start
					//of the array
					++accumulate_into;
					*accumulate_into = *it;
				}
			}

			//remove the unneeded entries.
			++accumulate_into;
			list.erase(accumulate_into, list.end());
		}

		static std::vector<Triplet> Merge(std::vector<Triplet>& left, std::vector<Triplet>& right) {
			//take two vectors and merge the right into the left, then compress
			//for fastest results, compress left and right ahead of time
			left.insert(left.end(), std::make_move_iterator(right.begin()), std::make_move_iterator(right.end()));
			Triplet::Compress(left);
			
			right.clear();
			right.shrink_to_fit();

			return left;
		}


		//////////////////////////////////////////////////////////////////////////////////////////////
		/// Static methods to extract blocks. This is useful when building hierarchical MG preconditioners.
		/// It is assumed that this is called from a single thread and after the full matrix was constructed.
		/// The input list may have its entries sorted (per-thread block) but will still be valid after this operation.
		///
		/// TODO: make a variant that takes advantage of list already being compressed and sorted.
		//////////////////////////////////////////////////////////////////////////////////////////////
		static std::vector<Triplet> RestrictToBlock(std::vector<Triplet>& list, StorageIndex i_lower, StorageIndex j_lower, StorageIndex i_upper, StorageIndex j_upper) {
			GUTIL_ASSERT(i_lower < i_upper && j_lower < j_upper);

			//each thread gets a contiguous region of list. The valid elements of that region will be compressed into the first k entries.
			//then each thread will (sequentially, in thread order) move its compressed data to the beginning of the list.
			std::vector<Triplet> result{};
			if (list.empty()) {return result;}

			//track the thread ranges for easier copying
			std::vector<gutil::OmpIteratorRange<decltype(list.begin())>> thread_ranges;
			std::vector<size_t> thread_size;

			auto in_block = [i_lower, j_lower, i_upper, j_upper](const Triplet& t) {
				return t.row() >= i_lower && t.row() < i_upper && t.col() >= j_lower && t.col() < j_upper;
			};

			//collect the COO data in the block
			GUTIL_OMP(parallel)
			{
				//copy per-thread range information
				gutil::OmpIteratorRange range(list.begin(), list.end());
				GUTIL_OMP(single)
				{
					thread_ranges.resize(range.n_threads);
					thread_size.resize(range.n_threads);
				}
				GUTIL_OMP(barrier)
				thread_ranges[range.tid] = range;

				//collect this thread's block coo data
				std::sort(range.begin, range.end);
				auto accumulate_into = range.begin;

				//check if we found a first valid entry
				bool found_first = false;
				for (auto it = range.begin; it!=range.end; ++it) {
					//increment it until it is in a valid block
					if (!in_block(*it)) {continue;}

					//ensure that the first accumulate entry is set correctly.
					if (!found_first) {
						*accumulate_into = *it;
						found_first = true;
						continue;
					}

					if (*it == *accumulate_into) {
						accumulate_into->val += it->val;
					}
					else {
						//keep the block to keep contiguous at the start
						//of the array
						++accumulate_into;
						*accumulate_into = *it;
					}
				}

				//track how many entries this thread has
				thread_size[range.tid] = found_first ? std::distance(range.begin, accumulate_into)+1 : 0;
			}

			//copy data and free the end of the list
			size_t total_size = 0;
			for (auto sz : thread_size) {
				total_size += sz;
			}

			result.reserve(total_size);
			for (auto range : thread_ranges) {
				result.insert(result.end(), range.begin, range.begin+thread_size[range.tid]);
			}

			return result;
		}



		//////////////////////////////////////////////////////////////////////////////////////////////
		/// Convert from a vector of triplets to vectors of i, j, and v (i.e., Eigen vs PETSc)
		//////////////////////////////////////////////////////////////////////////////////////////////
		static void SplitComponents(std::vector<StorageIndex>& coo_i, std::vector<StorageIndex>& coo_j, std::vector<Scalar_t>& coo_v, 
			std::vector<Triplet>&& coo_triplets) noexcept {
			GUTIL_ASSERT(coo_i.size()==coo_j.size() && coo_j.size()==coo_v.size());

			const size_t n = coo_triplets.size();
			const size_t offset = coo_i.size();
			
			coo_i.resize(coo_i.size() + n);
			coo_j.resize(coo_j.size() + n);
			coo_v.resize(coo_v.size() + n);

			GUTIL_OMP(parallel)
			{
				gutil::OmpIndexRange range(n);
				GUTIL_SIMD()
				for (size_t idx=range.begin; idx<range.end; ++idx) {
					coo_i[offset+idx] = coo_triplets[idx].row();
					coo_j[offset+idx] = coo_triplets[idx].col();
					coo_v[offset+idx] = coo_triplets[idx].val();
				}
			}

			//free old triplets
			coo_triplets.clear();
			coo_triplets.shrink_to_fit();
		}
	};


	////////////////////////////////////////////////////////////////////////
	/// Print triplets for debugging
	////////////////////////////////////////////////////////////////////////
	template<typename Scalar_t, typename StorageIndex=size_t, int StorageOrder=0> requires (StorageOrder==0 || StorageOrder==1)
	std::ostream& operator<<(std::ostream& os, const Triplet<Scalar_t,StorageIndex,StorageOrder>& triplet) noexcept {
		return os<<triplet.to_string();
	}
}


