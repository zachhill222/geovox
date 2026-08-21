#pragma once

#include "gutil.hpp"

#include <span>

namespace GV {
	///////////////////////////////////////////////////////////////////
	/// Caching classes to help evaluate dofs as needed and reduce
	/// redundant computations in combined kernels (e.g., K = K1 + 2*K2)
	///
	/// Similarly, use this to evaluate weighting functions once per element 
	/// rather than once per dof-pair
	///////////////////////////////////////////////////////////////////
	template<typename QuadRule_t>
	struct ScalarValueCache {
		

		///////////////////////////////////////////////////////////////
		/// Aliases, constants, primary storage and accessor
		///////////////////////////////////////////////////////////////
		using Scalar_t = typename QuadRule_t::Scalar_t;
		static constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

		Scalar_t val[N];

		[[nodiscard]] constexpr Scalar_t operator[](int n) const {
			GUTIL_ASSERT(n>=0 && n<N);
			return val[n];
		}
		[[nodiscard]] constexpr Scalar_t& operator[](int n) {
			GUTIL_ASSERT(n>=0 && n<N);
			return val[n];
		}

		static constexpr size_t size() noexcept {return static_cast<size_t>(N);}
		constexpr Scalar_t* data() noexcept {return val;}
		constexpr const Scalar_t* data() const noexcept {return val;}

		constexpr ScalarValueCache() : val{} {}
		constexpr ScalarValueCache(const ScalarValueCache&) noexcept = default;
		constexpr ScalarValueCache(ScalarValueCache&&) noexcept = default;
		constexpr ScalarValueCache& operator=(const ScalarValueCache&) noexcept = default;
		constexpr ScalarValueCache& operator=(ScalarValueCache&&) noexcept = default;

		static constexpr ScalarValueCache Filled(Scalar_t c) noexcept {
			ScalarValueCache result;
			std::fill(result.val, result.val + N, c);
			return result;
		}

		explicit ScalarValueCache(const Scalar_t* other_val) {
			GUTIL_ASSERT(other_val);
			GUTIL_SIMD()
			for (int i=0; i<N; ++i) {val[i]==other_val[i];}
		}

		///////////////////////////////////////////////////////////////
		/// Allow simple component-wise operations.
		///////////////////////////////////////////////////////////////
		constexpr ScalarValueCache& operator*=(Scalar_t a) noexcept {
			GUTIL_SIMD()
			for (int i=0; i<N; ++i) {val[i]*=a;}
			return *this;
		}
		constexpr ScalarValueCache& operator*=(const ScalarValueCache& other) noexcept {
			GUTIL_SIMD()
			for (int i=0; i<N; ++i) {val[i]*=other.val[i];}
			return *this;
		}
		constexpr ScalarValueCache& operator+=(const ScalarValueCache& other) noexcept {
			GUTIL_SIMD()
			for (int i=0; i<N; ++i) {val[i]+=other.val[i];}
			return *this;
		}

		///////////////////////////////////////////////////////////////
		/// Use constructors to do the primary evaluations.
		///////////////////////////////////////////////////////////////

		/// Initialzize to function values at the geometric (non-normalized)
		/// quadrature points.
		template<typename WeightFunc>
		ScalarValueCache(WeightFunc&& f, const QuadRule_t& qr) noexcept {
			f(std::span<Scalar_t,N>{val}, qr.geo_x(), qr.geo_y(), qr.geo_z());
		}

		/// Initialize to a scalar field. E.g. u(x,y,z) = sum_j c[j]*dof[j](x,y,z) (or any other meaningful linear combination);
		constexpr ScalarValueCache(std::span<const Scalar_t> coefs, std::span<const ScalarValueCache> dof_vals) noexcept : val{} {
			GUTIL_ASSERT(coefs.size()==dof_vals.size());
			for (size_t j=0; j<coefs.size(); ++j) {
				GUTIL_SIMD()
				for (int i=0; i<N; ++i) {
					val[i] += coefs[j]*dof_vals[j].val[i];
				}
			}
		}

		/// Initialize to a scalar field, but compute dof_vals on the fly
		template<typename DOF_t>
		constexpr ScalarValueCache(std::span<const Scalar_t> coefs, std::span<const DOF_t> dofs, const QuadRule_t& qr) : val{} {
			GUTIL_ASSERT(coefs.size()==dofs.size())
			for (size_t j=0; j<coefs.size(); ++j) {
				Scalar_t dof_vals[N];
				const uint8_t depth = dofs[j].depth_u8();
				const uint8_t loc   = dofs[j].local_dof_number(qr.support_element[depth]);
				auto qx=qr.quad_x(depth), qy=qr.quad_y(depth), qz=qr.quad_z(depth);
				dofs[j].evaluate_simd(loc, dof_vals, qx.data(), qy.data(), qz.data(), N);

				GUTIL_SIMD()
				for (int i=0; i<N; ++i) {
					val[i] += coefs[j]*dof_vals[i];
				}
			}
		}

		/// Initialize to a function of a pre-evaluated scalar field
		template<typename WeightFunc>
		constexpr ScalarValueCache(const ScalarValueCache& pre_eval, WeightFunc&& f, const QuadRule_t& qr) noexcept {
			GUTIL_SIMD()
			for (int i=0; i<N; ++i) {
				val[i] = f(pre_eval[i]);
			}
		}
	};


	template<typename QuadRule_t>
	ScalarValueCache<QuadRule_t> operator+(const ScalarValueCache<QuadRule_t>& left, const ScalarValueCache<QuadRule_t>& right) noexcept {
		ScalarValueCache<QuadRule_t> result{left};
		return result += right;
	}

	template<typename QuadRule_t>
	ScalarValueCache<QuadRule_t> operator*(const ScalarValueCache<QuadRule_t>& left, const ScalarValueCache<QuadRule_t>& right) noexcept {
		ScalarValueCache<QuadRule_t> result{left};
		return result *= right;
	}

	template<typename QuadRule_t>
	ScalarValueCache<QuadRule_t> operator*(const typename QuadRule_t::Scalar_t a, const ScalarValueCache<QuadRule_t>& right) noexcept {
		ScalarValueCache<QuadRule_t> result{right};
		return result *= a;
	}


	///////////////////////////////////////////////////////////////
	/// Cache vector values (e.g., gradients or matrix weights)
	///////////////////////////////////////////////////////////////
	template<typename QuadRule_t, int M=3>
	struct VectorValueCache {
		

		///////////////////////////////////////////////////////////////
		/// Aliases, constants, primary storage and accessor
		///////////////////////////////////////////////////////////////
		using Scalar_t = typename QuadRule_t::Scalar_t;
		static constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

		//layout as M blocks of size N (each block is one component)
		Scalar_t val[M*N];

		[[nodiscard]] Scalar_t* operator[](int i) noexcept {
			GUTIL_ASSERT(i<M);
			return val+N*i;
		}

		[[nodiscard]] const Scalar_t* operator[](int i) const noexcept {
			GUTIL_ASSERT(i<M);
			return val+N*i;
		}

		static constexpr size_t size() noexcept {return static_cast<size_t>(N);}
		constexpr Scalar_t* data() noexcept {return val;}

		constexpr VectorValueCache() : val{} {}
		constexpr VectorValueCache(const VectorValueCache&) noexcept = default;
		constexpr VectorValueCache(VectorValueCache&&) noexcept = default;
		constexpr VectorValueCache& operator=(const VectorValueCache&) noexcept = default;
		constexpr VectorValueCache& operator=(VectorValueCache&&) noexcept = default;

		///////////////////////////////////////////////////////////////
		/// Allow simple component-wise operations.
		///////////////////////////////////////////////////////////////
		VectorValueCache& operator*=(Scalar_t a) noexcept {
			GUTIL_SIMD()
			for (int i=0; i<M*N; ++i) {val[i]*=a;}
			return *this;
		}
		VectorValueCache& operator*=(const VectorValueCache& other) noexcept {
			GUTIL_SIMD()
			for (int i=0; i<M*N; ++i) {val[i]*=other.val[i];}
			return *this;
		}
		VectorValueCache& operator+=(const VectorValueCache& other) noexcept {
			GUTIL_SIMD()
			for (int i=0; i<M*N; ++i) {val[i]+=other.val[i];}
			return *this;
		}


		///////////////////////////////////////////////////////////////
		/// Use constructors to do the primary evaluations.
		///////////////////////////////////////////////////////////////

		/// Initialzize to function values at the geometric (non-normalized)
		/// quadrature points. Note that the function must be aware of the data layout
		template<typename WeightFunc>
		VectorValueCache(WeightFunc&& f, const QuadRule_t& qr) noexcept {
			f(std::span<Scalar_t,M*N>{val}, qr.geo_x(), qr.geo_y(), qr.geo_z());
		}
	};

	template<typename QuadRule_t, int M>
	VectorValueCache<QuadRule_t,M> operator+(const VectorValueCache<QuadRule_t,M>& left, const VectorValueCache<QuadRule_t,M>& right) noexcept {
		VectorValueCache<QuadRule_t,M> result{left};
		return result += right;
	}

	template<typename QuadRule_t, int M>
	VectorValueCache<QuadRule_t,M> operator*(const VectorValueCache<QuadRule_t,M>& left, const VectorValueCache<QuadRule_t,M>& right) noexcept {
		VectorValueCache<QuadRule_t,M> result{left};
		return result *= right;
	}

	template<typename QuadRule_t, int M>
	VectorValueCache<QuadRule_t,M> operator*(const typename QuadRule_t::Scalar_t a, const VectorValueCache<QuadRule_t,M>& right) noexcept {
		VectorValueCache<QuadRule_t,M> result{right};
		return result *= a;
	}


	///////////////////////////////////////////////////////////////////
	/// Utility operations between weights
	///////////////////////////////////////////////////////////////////
	template<typename QuadRule_t, int M>
	[[nodiscard]] inline constexpr ScalarValueCache<QuadRule_t> dot(const VectorValueCache<QuadRule_t,M>& left, const VectorValueCache<QuadRule_t,M>& right) noexcept {
		ScalarValueCache<QuadRule_t> result{};
		for (int j=0; j<M; ++j) {
			GUTIL_SIMD()
			for (int i=0; i<QuadRule_t::TOTAL_QUAD_POINTS; ++i) {
				result[i] += left[i][j]*right[i][j];
			}
		}
		return result;
	}


	template<typename QuadRule_t, int M>
	[[nodiscard]] inline constexpr ScalarValueCache<QuadRule_t> component_sum(const VectorValueCache<QuadRule_t,M>& left) noexcept {
		ScalarValueCache<QuadRule_t> result{};
		for (int j=0; j<M; ++j) {
			GUTIL_SIMD()
			for (int i=0; i<QuadRule_t::TOTAL_QUAD_POINTS; ++i) {
				result[i] += left[i][j];
			}
		}
		return result;
	}


	///////////////////////////////////////////////////////////////////
	/// For dof values specifically, we may need to record the depth
	/// so that the correct jacobian information can be looked up.
	///////////////////////////////////////////////////////////////////
	template<typename QuadRule_t>
	struct DofValueCache : ScalarValueCache<QuadRule_t> {
		using BASE = ScalarValueCache<QuadRule_t>;
		using BASE::N;
		using BASE::val;
		const uint8_t depth;

		/// Initalize by evaluating a dof
		template<typename DOF_t>
		DofValueCache(DOF_t dof, const QuadRule_t& qr) : depth(dof.depth_u8()) {
			const uint8_t loc  = dof.local_dof_number(qr.support_element[depth]);
			auto qx=qr.quad_x(depth), qy=qr.quad_y(depth), qz=qr.quad_z(depth);
			dof.evaluate_simd(loc, val, qx.data(), qy.data(), qz.data(), N);
		}
	};


	template<typename QuadRule_t>
	struct DofGradCache : VectorValueCache<QuadRule_t,3> {
		using BASE = VectorValueCache<QuadRule_t,3>;
		using BASE::N;
		using BASE::val;
		const uint8_t depth;

		template<typename DOF_t>
		DofGradCache(DOF_t dof, const QuadRule_t& qr) : depth(dof.depth_u8()) {
			const uint8_t loc  = dof.local_dof_number(qr.support_element[depth]);
			auto qx=qr.quad_x(depth), qy=qr.quad_y(depth), qz=qr.quad_z(depth);
			dof.gradient_simd(loc, val, val+N, val+2*N, qx.data(), qy.data(), qz.data(), N);
		}
	};


	//////////////////////////////////////////////////////////////////
	/// When using diffuse domain methods, we may need the signed distance
	/// at each quadrature point multiple times. Each evaluation may be
	/// expensive and require an octree lookup of a nearest object.
	/// Thus, it is best to cache the sdf values and re-use them when possible.
	///////////////////////////////////////////////////////////////////
	template<typename QuadRule_t>
	struct SdfValueCache :  public ScalarValueCache<QuadRule_t> {
		using Scalar_t = typename QuadRule_t::Scalar_t;
		using BASE = ScalarValueCache<QuadRule_t>;
		using BASE::BASE;
		using BASE::N;
		using BASE::val;

		template<typename Assembly_t>
		SdfValueCache(const Assembly_t& assembly, const QuadRule_t& qr) {
			static_assert( std::same_as<typename QuadRule_t::Scalar_t, typename Assembly_t::Scalar_t> );
			assembly.signed_distance({val, N}, qr.geo_x(), qr.geo_y(), qr.geo_z());
		}
	};

	template<typename QuadRule_t>
	struct SdfGradCache :  public VectorValueCache<QuadRule_t> {
		using Scalar_t = typename QuadRule_t::Scalar_t;
		using BASE = VectorValueCache<QuadRule_t>;
		using BASE::BASE;
		using BASE::N;
		using BASE::val;

		template<typename Assembly_t>
		SdfGradCache(const Assembly_t& assembly, const QuadRule_t& qr) {
			static_assert( std::same_as<typename QuadRule_t::Scalar_t, typename Assembly_t::Scalar_t> );
			assembly.grad_signed_distance({val, 3*N}, qr.geo_x(), qr.geo_y(), qr.geo_z());
		}
	};


	///////////////////////////////////////////////////////////////////
	/// When computing values of a linear or bilinear form, it is best to 
	/// pre-compute values per-dof and cache the values rather than compute 
	/// values per dof pair on the fly. This is more applicable to bilinear forms,
	/// but logic can be re-used for linear forms.
	///
	/// This is intended to be created once per thread and then updated once per element.
	///////////////////////////////////////////////////////////////////
	template<typename Kernel_t, typename DofHandler_t, typename QuadRule_t>
	struct ElementDofCache {


		///////////////////////////////////////////////////////////////
		/// Aliases and data
		///////////////////////////////////////////////////////////////
		using DOF_t      = typename DofHandler_t::DOF_t;
		using MeshElem_t = typename DofHandler_t::Mesh_t::Elem_t;
		using Scalar_t   = typename QuadRule_t::Scalar_t;

		using DofValueCache_t = DofValueCache<QuadRule_t>;
		using DofGradCache_t  = DofGradCache<QuadRule_t>;

		std::vector<DOF_t>  dofs;		//active dofs on the current element
		std::vector<size_t> global_idx;		//global numbers of the dofs
		
		std::vector<DofValueCache_t> vals;
		std::vector<DofGradCache_t>  grad;

		const DofHandler_t& handler;
		const QuadRule_t& 	qr;


		///////////////////////////////////////////////////////////////
		/// Constructor to link handler and quadrature rule
		///////////////////////////////////////////////////////////////
		ElementDofCache(const DofHandler_t& h, const QuadRule_t& q) : handler(h), qr(q) {}
		[[nodiscard]] size_t size() const noexcept {return dofs.size();}

		///////////////////////////////////////////////////////////////
		/// Primary method for caching values. The quadrature rule must already
		/// be updated.
		///////////////////////////////////////////////////////////////
		void gather_qh() noexcept {
			dofs = handler.get_active_dofs_quasi_hierarchical(qr.q_el);
			const size_t n = dofs.size();

			global_idx.resize(n);
			if constexpr (Kernel_t::NEEDS_DOF_VALS) {vals.clear(); vals.reserve(n);}
			if constexpr (Kernel_t::NEEDS_DOF_GRAD) {grad.clear(); grad.reserve(n);}

			for (size_t j=0; j<n; ++j) {
				global_idx[j] = handler.global_number(dofs[j]);
				GUTIL_ASSERT(global_idx[j] < handler.n_dofs());
				GUTIL_ASSERT(dofs[j].depth() <= qr.q_el.depth()+1);	//for quasi-hierarchical refinement
				if constexpr (Kernel_t::NEEDS_DOF_VALS) {vals.emplace_back(dofs[j], qr);}
				if constexpr (Kernel_t::NEEDS_DOF_GRAD) {grad.emplace_back(dofs[j], qr);}
			}
		}


		///////////////////////////////////////////////////////////////
		/// A few query methods to help with computing weights
		///////////////////////////////////////////////////////////////
		[[nodiscard]] ScalarValueCache<QuadRule_t> reconstruct_field(std::span<const Scalar_t> global_coefs) noexcept {
			GUTIL_ASSERT(!vals.empty() && "the dof values were not cached");
			GUTIL_ASSERT(handler.n_dofs() == global_coefs.size());
			GUTIL_ASSERT(vals.size() == dofs.size());

			//gather local coefficients
			ScalarValueCache<QuadRule_t> result{};
			for (size_t j=0; j<dofs.size(); ++j) {
				result += global_coefs[global_idx[j]] * vals[j];
			}

			//evaluate and return
			return result;
		}
	};


}