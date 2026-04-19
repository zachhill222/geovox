#pragma once

#include<vector>
#include<algorithm>
#include<numeric>
#include<memory_resource> //TODO: arena allocating didn't seem to do much. think about this.
#include<Eigen/SparseCore>
#include "util/log_time.hpp"

namespace GV
{
	//In order to allow DOF renumbering and excessive lookups, we need to compute permutation arrays
	//to convert an arbitrary order of row/column selections into the standard order
	template<typename Key_t>
	static constexpr std::vector<uint64_t> get_sort_perm(const std::vector<Key_t>& select) {
		std::vector<uint64_t> P(select.size());
		std::iota(P.begin(), P.end(), 0); //set P=[0, 1, 2, ...]

		//create a comparison to sort P
		auto comp = [&select](uint64_t a, uint64_t b) {return select[a] < select[b];};
		std::sort(P.begin(), P.end(), comp);
		return P;
	}

	//each bilinear form will be represented by a matrix (TODO: add matrix free vector operations)
	//the matrix can be incrementally updated between mesh refinements (TODO: add this)
	//because of this, the bilinear form holds data for the matrix
	template<typename ColKey_t=uint64_t>
	struct SparseRowEntry
	{
		ColKey_t col;
		double   val;
		SparseRowEntry(ColKey_t col, double val) : col(col), val(val) {}

		//comparisons for sorting the row by index
		inline constexpr bool operator==(const SparseRowEntry& other) const {
			return col==other.col;
		}

		inline constexpr auto operator<(const SparseRowEntry& other) const {
			return col<other.col;
		}

		inline SparseRowEntry& operator+=(const SparseRowEntry& other) {
			assert(col==other.col);
			val += other.val;
			return *this;
		}

		inline SparseRowEntry& operator+=(const double inc) {
			val += inc;
			return *this;
		}

		inline SparseRowEntry& operator*=(const double inc) {
			val *= inc;
			return *this;
		}
	};


	//struct for storing rows of the Pseudo CSC matrix.
	template<typename RowKey_t = uint64_t, typename ColKey_t=uint64_t>
	struct SparseRow
	{
		using Entry_t = SparseRowEntry<ColKey_t>;
		using Iter_t  = std::pmr::vector<Entry_t>::iterator;
		using CIter_t = std::pmr::vector<Entry_t>::const_iterator;

		RowKey_t row_id;
		std::pmr::vector<Entry_t> entries; 
		SparseRow() : row_id{} {}
		explicit SparseRow( const RowKey_t r_id, 
							uint64_t nz=0,
							std::pmr::memory_resource* mr = std::pmr::get_default_resource() //use arena/pool allocators
							) : row_id{r_id}, entries{mr} {entries.reserve(nz);}
		
		//for accessing, the entries must be sorted
		inline void sort() {std::sort(entries.begin(), entries.end());}

		inline CIter_t lower_bound(const Entry_t& entry) const {
			return std::lower_bound(entries.begin(), entries.end(), entry);
		}

		inline Iter_t lower_bound(const Entry_t& entry) {
			return std::lower_bound(entries.begin(), entries.end(), entry);
		}

		double value(const ColKey_t col) const {
			CIter_t it = lower_bound(Entry_t{col,0.0});
			return (it==entries.end() || it->col != col) ? 0.0 : it->val;
		}

		uint64_t row_nnz(const std::vector<ColKey_t>& col_select, 	//the requested columns (may or may not exist, arbitrary order, no duplicates)
					const std::vector<uint64_t>& perm				//the permutation vector such that col_select[perm[k]] < col_select[perm[k+1]]
					) const {

			//start at the beginning of this row in standard order
			CIter_t c_it = entries.begin();
			uint64_t p = 0;

			//number of stored entries
			uint64_t r_nnz = 0;

			//loop through the permutation vector and synchronize the iterators
			const uint64_t cs_size = col_select.size();
			while ( (c_it!=entries.end()) && (p<cs_size) ) {
				const uint64_t col_select_idx = perm[p];
				const ColKey_t c_k = col_select[col_select_idx];

				//increase c_it until it reaches or goes past c_k
				if      (c_it->col < c_k) {++c_it;} //haven't found the desired column
				else if (c_it->col > c_k) {++p;}    //we found and dealt with the desired column or it didn't exist
				else {
					//we are at the desired column and it exists
					++r_nnz;
					++p; ++c_it;
				}
			}

			return r_nnz;
		}


		//find the values at the requested columns provided the permutation vector
		//results are appended to the supplied nz_idx and nv_vals. This is intended to be called as part of the 
		//actual csr matrix assembly. The offset should be the number of non-zero entries BEFORE this row. 
		//
		//If A is the csr matrix, then we need A(r,c) (where r is this row) to satisfy
		//A(r,c) = nv_vals[k] where c = col_idx[k] and RowOffset[r]<= k < RowOffset[r+1].
		//
		//note that it is best to find the requested values in the standard order, but they must be put into
		//col_idx and nv_vals in the provided order. To do this, we make a first pass to compute the number of nonzero entries per row.
		//then this method populates the column index and values in the requested order
		//the random access iterators into where the column and values are being stored must be provided (both start and end for this row)
		//
		//Note that this method is thread safe during the second pass (as long as two threads don't have the same row)
		template<typename ColIdx_iter, typename ValIdx_iter>
		void csr_v_idx(const std::vector<ColKey_t>& col_select, 		//the requested columns (may or may not exist, arbitrary order, no duplicates)
					const std::vector<uint64_t>& perm,					//the permutation vector such that col_select[perm[k]] < col_select[perm[k+1]]
					ColIdx_iter c_idx_start, ColIdx_iter c_idx_end,		//range into the column index vector that belongs to this row (half open)
					ValIdx_iter vals_start, ValIdx_iter vals_end)		//range into the values corresponding to the column index. (half open)
					const {

			//verify that the number of entries for the column index and values are the same
			assert(std::distance(c_idx_start, c_idx_end) == std::distance(vals_start, vals_end));

			//start at the beginning of this row in standard order
			CIter_t c_it = entries.begin();
			uint64_t p = 0;

			//loop through the permutation vector and synchronize the iterators
			const uint64_t cs_size = col_select.size();
			while ( (c_it!=entries.end()) && (p<cs_size) && (c_idx_start!=c_idx_end) && (vals_start!=vals_end) ) {
				const uint64_t col_select_idx = perm[p];
				const ColKey_t c_k = col_select[col_select_idx];

				//increase c_it until it reaches or goes past c_k
				if      (c_it->col < c_k) {++c_it;} //haven't found the desired column
				else if (c_it->col > c_k) {++p;}    //we found and dealt with the desired column or it didn't exist
				else {
					//we are at the desired column and it exists
					*c_idx_start = col_select_idx;
					*vals_start = c_it->val;

					++c_idx_start; ++vals_start;
					++p; ++c_it;
				}
			}

			//verify that we actually found all the expected entries
			assert(c_idx_start==c_idx_end);
			assert(vals_start==vals_end);
		}

		//insert and preserve sorting
		void insert(Entry_t&& entry) {
			Iter_t it = lower_bound(entry);
			if (it==entries.end()) {
				//new entry at end
				entries.push_back(std::move(entry));
			}
			else if (it->col == entry.col) {
				//accumulate existing entry
				it->val += entry.val;
			}
			else {
				//new interior entry (try to avoid this)
				entries.insert(it, std::move(entry));
			}
		}

		//forward common vector operations
		inline void push_back(Entry_t&& entry) {entries.push_back(std::move(entry));}
		inline void push_back(Entry_t entry) {entries.push_back(entry);}
		inline void emplace_back(ColKey_t col, double val) {entries.emplace_back(col,val);}
		inline uint64_t size() const {return entries.size();}
		inline void shrink_to_fit() {entries.shrink_to_fit();}
		inline void reserve(const uint64_t sz) {entries.reserve(sz);}


		//we can add entries to the end and then accumulate all at once
		//TODO: write binary version (accumulate left/right and then join)
		void accumulate() {
			if (entries.empty()) {return;}

			std::sort(entries.begin(), entries.end());
			Iter_t it = entries.begin(); //write iterator
			for (Iter_t cur = it+1; cur != entries.end(); ++cur) {
				if (cur->col == it->col) {
					it->val  += cur->val;
					cur->val = 0.0;
				}
				else {
					++it; //done writing to this spot
					*it = *cur; //copy the read iterator into the next contiguous memory location
				}
			}

			entries.erase(it+1, entries.end());
		}

		//comparison for sorting rows by their id
		constexpr bool operator==(const SparseRow& other) const {return row_id == other.row_id;}
		constexpr bool operator==(const RowKey_t& r_id) const {return row_id < r_id;}

		//add two rows with the same id
		//this can be used as either a row sum (two different row ids) or as an accumulation
		//of a temporary row with the primary row in the csc matrix
		//the row will need to be accumulated/compressed after this
		inline void append(SparseRow<RowKey_t,ColKey_t>&& other) {
			entries.insert(entries.end(),
				std::make_move_iterator(other.entries.begin()),
				std::make_move_iterator(other.entries.end()));
		}
	};

	//for sorting rows, we need non-member comparisons to row_id and the row itself (otherwise we have to use a proxy row object)
	template<typename RowKey_t, typename ColKey_t>
	inline constexpr bool operator<(const RowKey_t& r_id, const SparseRow<RowKey_t,ColKey_t>& row) {return r_id < row.row_id;}

	template<typename RowKey_t, typename ColKey_t>
	inline constexpr bool operator>(const RowKey_t& r_id, const SparseRow<RowKey_t,ColKey_t>& row) {return r_id > row.row_id;}

	// template<typename RowKey_t, typename ColKey_t>
	// inline constexpr bool operator==(const SparseRow<RowKey_t,ColKey_t>& row, const RowKey_t& r_id) {return r_id == row.row_id;}
	
	template<typename RowKey_t, typename ColKey_t>
	inline constexpr bool operator<(const SparseRow<RowKey_t,ColKey_t>& row, const RowKey_t& r_id) {return row.row_id < r_id;}

	template<typename RowKey_t, typename ColKey_t>
	inline constexpr bool operator>(const SparseRow<RowKey_t,ColKey_t>& row, const RowKey_t& r_id) {return row.row_id > r_id;}


	//Pseudo CSR structure - essentially COO in disguise
	//use as a starting point to build an Eigen CSR matrix
	//rows can be sorted by their RowKey_t (e.g, a uint64_t or similar such as a DOF or mesh feature key)
	//
	//The intended use is for each row i (test) dof, we have a SparseRow that records the non-zero values
	//of integral_D a(test_dof_i, trial_dof_j) over the entire domain.
	//During refinement and coarsening, we may not wish to lose this information. Additionally, which dofs
	//are active will change frequently. Rather than storing row information by a compressed dof number
	//(which is required for the linear solve), we index rows by the dof key (still effectively an integer).
	//Then as new dofs are activated, their row can be added and their contribution to the other rows more easily
	//incremented. Similarly, the column (trial) dofs can act as the index into each sparse row.
	template<typename RowKey_t=uint64_t, typename ColKey_t=uint64_t>
	struct CSR_COO
	{
		std::pmr::monotonic_buffer_resource arena;

		std::pmr::vector<SparseRow<RowKey_t,ColKey_t>> rows;
		using Row_t      = SparseRow<RowKey_t,ColKey_t>;
		using Entry_t    = typename Row_t::Entry_t;
		using CRowIter_t = std::pmr::vector<Row_t>::const_iterator;
		using RowIter_t  = std::pmr::vector<Row_t>::iterator;

		explicit CSR_COO(const uint64_t initial_size = 1<<20) : arena{initial_size}, rows{&arena} {}

		//accumulate and compress the rows
		void accumulate_each_row() {
			for_each_row_omp([this](Row_t& r){r.accumulate();});
		}

		//perform actions on the rows either in serial or parallel
		template<typename Action>
		void for_each_row_omp(Action&& action) {
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (uint64_t r=0; r<rows.size(); ++r) {
				action(rows[r]);
			}
		}

		template<typename Action>
		void for_each_row(Action&& action) {
			for (uint64_t r=0; r<rows.size(); ++r) {
				action(rows[r]);
			}
		}

		//sort the rows by their index
		inline void sort_rows() {
			std::sort(rows.begin(), rows.end());
		}

		//search for a row by its row ID
		inline CRowIter_t lower_bound(const RowKey_t r_id) const {
			return std::lower_bound(rows.begin(), rows.end(), r_id);
		}

		inline RowIter_t lower_bound(const RowKey_t r_id) {
			return std::lower_bound(rows.begin(), rows.end(), r_id);
		}

		//access a current row or create a new row and return a reference
		//the rows must be sorted for this to work. This is not always thread safe.
		//we assume the rows are sorted and preserve this.
		Row_t& get_row(const RowKey_t r_id) {
			RowIter_t it = lower_bound(r_id);
			if (it==rows.end()) {
				rows.emplace_back(r_id, 0, &arena);
				return rows.back();
			}
			
			if (it->row_id != r_id) {
				return *rows.insert(it, Row_t{r_id, 0, &arena});
			}
			else {
				return *it;
			}
		}

		const Row_t& get_row(const RowKey_t r_id) const {
			CRowIter_t it = lower_bound(r_id);
			if ( (it!=rows.end()) && (it->row_id==r_id)) {
				return *it;
			}
			throw std::runtime_error("CSR_COO::get_row (const) - row not found");
		}

		//if we know that a new row needs to be created or are ok with
		//accumulating two rows with the same id later, we can quickly append a new row
		//and return a reference to it
		Row_t& new_row(const RowKey_t r_id) {
			rows.emplace_back(r_id, 0, &arena);
			return rows.back();
		}


		//when many new rows have been added, we will need to combine them with any existing
		//rows that have the same id and ensure that they are sorted for lookups
		void join_rows() {
			if (rows.empty()) {return;}

			sort_rows();
			RowIter_t it = rows.begin(); //write iterator
			for (RowIter_t cur = it+1; cur != rows.end(); ++cur) {
				if (cur->row_id == it->row_id) {
					it->append(*cur);
				}
				else {
					++it; //done writing to this spot
					std::swap(*it,*cur); //copy the read iterator into the next contiguous memory location
				}
			}

			rows.erase(it+1, rows.end());
		}

		//accumulate all rows together to put the CSC into a valid state
		void accumulate() {
			join_rows();
			accumulate_each_row();
		}

		//remove extra reserved space
		void shrink_to_fit() {
			rows.shrink_to_fit();
			for_each_row_omp([this](Row_t& row) {row.shrink_to_fit();});
		}


		//given a selection of row and column dofs, build the Eigen CSR matrix
		//the structure must be in a valid state. call accumulate() first.
		//any re-numbering of the DOFs occur in the ordering of row_select and col_select.
		//row_select and col_select do not need to be sorted, but they must not contain
		//duplicate entries. We also assume that each requested row corresponds to one (and only one) existing row
		//that is stored here and that for each row, there is at least one requested column that has a stored entry.

		//Note Eigen sparse matrices must have signed index type. https://libeigen.gitlab.io/eigen/docs-5.0/classEigen_1_1SparseMatrix.html
		Eigen::SparseMatrix<double,Eigen::RowMajor,int> to_eigen_csr(
			const std::vector<RowKey_t>& row_select, 
			const std::vector<ColKey_t>& col_select) const;
	};


	template<typename RowKey_t, typename ColKey_t>
	Eigen::SparseMatrix<double,Eigen::RowMajor,int> CSR_COO<RowKey_t,ColKey_t>::to_eigen_csr(
		const std::vector<RowKey_t>& row_select, 
		const std::vector<ColKey_t>& col_select) const
	{
		LogTime t0{"build csr"};
		const int n_rows = static_cast<int>(row_select.size());
		const int n_cols = static_cast<int>(col_select.size());

		//Because row/col_select are allowed to be in arbitrary order, we compute permutations
		//that would put them into the standard order so that we can better look up values in the COO_CSR structure
		const auto PR = get_sort_perm(row_select);
		const auto PC = get_sort_perm(col_select);

		//initialize the matrix
		Eigen::SparseMatrix<double,Eigen::RowMajor,int> mat(n_rows,n_cols);

		//recall that CSR matrices have the storage format:
		//
		//		row offset array : RO = {a_0, a_1, a_2, ... , a_{n_rows}}
		//		column indices	 : CI = {b_0, b_1, b_2, b_3, b_4, ... , b_{nnz-1}}
		//		values           :  V = {v_0, v_1, v_2, v_3, v_4, ... , v_{nnz-1}}
		//
		//where the indices into both CI and V for row r are the half open interval [RO[r], RO[r+1]).
		//To read the matrix value at A(r,c), we first find k in [RO[r],RO[r+1]) such that CI[k]=c.
		//If such a k exists, then A(r,c) = V[k]. If such a k does not exist, the entry is not stored
		//and A(r,c)=0.
		
		//Note that RO is by far the smallest array. We write to the matrix in two passes.
		//First, build RO and count the total number of nonzero values. Then we populate CI and V.
		
		//Build RO (Eigen outerIndexPtr when in Sparse RowMajor)
		//Eigen reserves enough space when resizing to n_rows as RowMajor
		//see around line 750 (April 15, 2026): https://libeigen.gitlab.io/eigen/docs-5.0/SparseMatrix_8h_source.html
		assert(mat.outerSize() == n_rows); //If Eigen changes behavior, hopefully this will fail.
		int* RO = mat.outerIndexPtr(); 
		std::fill(RO, RO+n_rows+1, 0); //I think this is redundant, but would be uncaught if Eigen changes.
		
		//loop through the row permutation and synchronize the global/row keys
		LogTime* t1 = new LogTime{"row offsets"};

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int r=0; r<n_rows; ++r) {
			CRowIter_t it = lower_bound(row_select[r]);
			if (it != rows.end() && it->row_id == row_select[r]) {
				RO[r+1] = it->row_nnz(col_select, PC);
			}
		}



		//only the number of nonzero entries are stored for each row. the offsets must be accumulated.
		for (int r=0; r<n_rows; ++r) {
			RO[r+1] += RO[r];
		}
		delete t1;
		
		//the offsets are and number of nonzeros are known. initialize the matrix and build the inner(column) and value arrays
		LogTime* t2 = new LogTime{"reserve, populate col and val"};
		int nnz = RO[n_rows];
		mat.resizeNonZeros(nnz);

		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int r=0; r<n_rows; ++r) {
			auto c_start = mat.innerIndexPtr() + mat.outerIndexPtr()[r];
			auto c_end   = mat.innerIndexPtr() + mat.outerIndexPtr()[r+1];
			auto v_start = mat.valuePtr()	   + mat.outerIndexPtr()[r];
			auto v_end   = mat.valuePtr()      + mat.outerIndexPtr()[r+1];

			const RowKey_t r_k = row_select[r];
			CRowIter_t r_it = lower_bound(r_k);
			if ( (r_it==rows.end()) || (r_it->row_id != r_k) ) {assert(false); continue;} //TODO: terminate? the matrix has a zero row which is probably wrong.

			r_it->csr_v_idx(col_select, PC, c_start, c_end, v_start, v_end);
		}
		delete t2;

		return mat;
	}
}