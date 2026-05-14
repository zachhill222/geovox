#pragma once

#include "fem/numerics/quad_point_map.hpp"
#include "util/log_time.hpp"

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <barrier>
#include <chrono>

#include <type_traits>
#include <cstdlib>
#include <tuple>
#include <utility>
#include <span>

namespace GV
{
	//A generic kernel type for assembline FEM matrices.
	//This class will construct one or more local matricies (e.g., mass/stiffness)
	//over a single quadrature element of m trial basis functions (solution dofs) against
	//n test basis functions (also dofs, but can be of a different type than the trial).
	//the result is an n-by-m local matrix (stored as a vector).
	//This is designed for use in the CHARMS method, so m and n are unknown at compile time.
	//The H in CHARMS stands for Hierarchical. For each quadrature point in the quadrature element,
	//we must `project' it to the reference coordinate in the support element for each basis function.
	//This only needs to be done once for each unique depth of the basis functions (in the mesh/function space hierarchy)
	//To avoid doing this computation (and to avoid unnecessary looping through the mesh elements) more than necessary,
	//multiple evaluation methods can be passed to the kernel. One local matrix is produced per evaluation method.
	//The evaluation method must take the signature
	//
	//		static constexpr void eval(val,Jxx,Jyy,Jzz, psi_i,spt_i,X_i,Y_i,Z_i, phi_j,spt_j,X_j,Y_j,Z_j);
	//
	//Where val is a reference to an (un-initialized) array<double,NQ> to store the values, J** are the diagonal entries of the jacobian,
	//psi_i is the test dof, phi_j is the trial dof, X_*, Y_*, Z_* are const references to an array<double,NQ>
	//that record the reference coordinate of basis psi_i or phi_j on its support element spt_*.
	//The DOF types should provide vectorized methods for eval or grad as needed with a similar signature.
	//Because gradients are present in some bilenear forms and not others, it is the eval's responsibiilty
	//to handle the jacobian. this includes the final multiply by the determinant.
	//
	//For a bilinear form integral_D a(psi, phi) dx, the corresponding entry in the global matrix is 
	//		A_IJ = integral_D a(psi_I,phi_J) = sum_E integral_E a(psi_I,phi_J)
	//where D is the problem domain, a(*,*) is the bilinear form, and E is an element of the mesh of D.
	//The local matrix for element E is then
	//      a_ij = integral_E a(psi_I, phi_J) ~ sum_q a(psi_i, phi_j, q) * w_q
	//where i and j are local dof numbers corresponding to the global numbers I and J, q is a quadrature point
	//with w(q) its corresponding quadrature weight. For hierarchical methods, it is more feasible to convert
	//the quadrature points to the correct reference points for each function psi_i and phi_j ahead of time.
	//the supplied eval(*) method described above is responsible for evaluating a(psi_i, phi_j, q) at each of the
	//supplied quadrature points. The values are then reduced with multiplication with the wieights into the local a_ij entry.
	//
	//Note that it is the responsibility of the evaluator method to correctly handle the jacobian.
	//The jacobian depends only on the mapping from the reference element to the quadrature element.
	//For an octree voxel mesh, this depends only on the depth of the quadrature element and the dimensions of the domain.


	//A container to hold a form and thread so that each thread can be dispatched concurrently
	template<typename Form_t, typename QuadRule_t>
	struct FormThread {
		//logic for synchronizing individual dispatches (prepare, compute, finalize, scatter operations)
		std::thread 			thread;
		mutable std::mutex 		mtx;
		std::condition_variable cv;
		bool					ready = false;		//we can dispatch it
		bool					running = false;	//we have dispatched it and it is working
		bool 					shutdown = false;	//exit the dispatch-wait loop
		bool					linked_to_kernel = false;	//ensure that the kernel data is accessible

		//logic for synchonizing all dispatches for many forms
		std::barrier<>*			sync{nullptr};	//increment a counter in the main thread

		//link to the form and kernel data needed to pass to the form
		Form_t& form;
		QuadRule_t const* q_map{nullptr};

		//link to a kernel
		void link_kernel(std::barrier<>& k_sync, const QuadRule_t& k_q_map) {
			if (linked_to_kernel) {throw std::runtime_error("FormThread - Already linked to a kernel");}

			sync             = &k_sync; //must be mutable
			q_map		     = &k_q_map;
			linked_to_kernel = true;
		}

		//constructor to link to main thread and form resources
		FormThread(Form_t& form_) : form(form_) {
			shutdown = false;
			thread = std::thread([this]{dispatch_loop();});
		}

		//we can't move or copy this class as it owns a thread
		FormThread(const FormThread&) 			 = delete;
		FormThread(FormThread&&) 				 = delete;
		FormThread& operator=(const FormThread&) = delete;
		FormThread& operator=(FormThread&&) 	 = delete;

		//ensure that the thread is joined before this is destoyed
		~FormThread() {
			stop();
			thread.join();
		}

		//call to allow the dispatch_loop to continue
		inline void dispatch() {
			{
				std::lock_guard lock(mtx);
				ready = true;
			}
			cv.notify_one();
		}

		//call to stop the dispatch_loop
		inline void stop() {
			{
				std::lock_guard lock(mtx);
				shutdown = true;
			}
			cv.notify_one();
		}

		void dispatch_loop() {
			while (true) {
				std::unique_lock lock(mtx);
				//note cv.wait releases the lock until the cv is notified and the predicate returns true
				//the thread blocks here (not spinning) until the cv is notified and the predicate returns true
				cv.wait(lock, [this]{return ready || shutdown;});
				
				if (shutdown) {return;}
				assert(linked_to_kernel);

				ready   = false;
				form.prepare();
				form.compute(*q_map);
				form.finalize();
				form.scatter();

				//tell the kernel that we finished
				sync->arrive_and_wait();
			}
		}
	};



	//Kernel class allows multiple interactions (bilinear forms) to be integrated simultaneously
	//The bilinear forms in the kernel are allowed to have different dof types
	//but they must all have compatable quadrature element types (i.e., voxel elements with the same maximum depth)
	//the kernel accepts vectors of dofs (essentially uint64_t with additional logic), a quadrature element
	//and then organizes the projection of the quadrature points from the quadrature element into the support elements
	//of the dofs. Then it dispatches the dofs to appropriate bilinear forms and accumulates the results against the
	//quadrature weights into the local matrix.
	template<uint64_t N_QUAD_POINTS, typename... Form_ts>
	struct Kernel
	{
		//organize forms and collect types
		static constexpr uint64_t N_FORMS   = sizeof...(Form_ts);
		static_assert(N_FORMS>0, "Kernel - no form was provided");

		//get the quadrature element type
		using QuadElem_t = typename std::tuple_element_t<0, std::tuple<Form_ts...>>::QuadElem_t;
		using QuadRule_t = QuadPointMap<QuadElem_t,N_QUAD_POINTS>;

		static_assert((std::same_as<typename Form_ts::QuadElem_t, QuadElem_t> && ...),
			"Kernel - all forms must share the same type of quadrature element (QuadElem_t).");

		//access individual forms
		template<int I>
		auto& form() {return std::get<I>(Threads).form;}
		template<int I>
		const auto& form() const {return std::get<I>(Threads).form;}

		Kernel(	Form_ts&... Forms) : Threads(Forms...) {
			//let the quadrature class collect the mesh extents to compute jacobians
			q_map.set_bounds(form<0>().trial_handler.mesh.low, form<0>().trial_handler.mesh.high);
			
			//link kernel to all the threads
			std::apply([this](auto&... threads){ (threads.link_kernel(sync, q_map), ...);}, Threads);
		}

		~Kernel() {
			std::apply([](auto&... threads){ (threads.stop(), ...);}, Threads);
		}

		//the kernel owns threads and cannot be moved/copied
		Kernel(const Kernel&) 			 = delete;
		Kernel(Kernel&&) 				 = delete;
		Kernel& operator=(const Kernel&) = delete;
		Kernel& operator=(Kernel&&)      = delete;

		//interface to use in the element loop
		//note that set_basis must be called on each form individually
		inline void set_element(QuadElem_t el) {q_map.set_quad_element(el);}

		//main dispatch loop
		void dispatch_all() {
			std::apply([](auto&... threads){ (threads.dispatch(), ...);}, Threads);
			sync.arrive_and_wait();
		}


	private:
		//quadrature data
		QuadRule_t q_map;
		
		//pair a new thread to each form
		std::tuple<FormThread<Form_ts, QuadRule_t>...> Threads;

		//synchronization data
		std::barrier<> sync{N_FORMS+1};
	};
	

	// template<uint64_t N_QUAD_POINTS, typename... BiLinearForms_ts, typename... LinearForms_ts>
	// template<uint64_t I>
	// void Kernel<N_QUAD_POINTS, TypeList<BiLinearForms_ts...>, TypeList<LinearForms_ts...>>::L_compute()
	// {
	// 	static_assert(requires {
	// 		std::declval<const L_Form<I>&>().eval(
	// 			std::declval<std::array<double,NQ>&>(),
	// 			std::declval<double>(),
	// 			std::declval<double>(),
	// 			std::declval<double>(),
	// 			std::declval<const typename L_Form<I>::TestDOF_t>(),
	// 			std::declval<const QuadElem_t>(),
	// 			std::declval<const std::array<double,NQ>&>(),
	// 			std::declval<const std::array<double,NQ>&>(),
	// 			std::declval<const std::array<double,NQ>&>()
	// 			);
	// 		}, "Kernel - LinearForm does not have an eval() method with the required signature.");

	// 	const uint64_t n_test=L_form<I>().n_test;

	// 	#ifdef _OPENMP
	// 	#pragma omp parallel if(n_test > KERNEL_OMP__BASIS_THRESHOLD)
	// 	#endif
	// 	{
	// 		std::array<double,NQ> vals;
	// 		#ifdef _OPENMP
	// 		#pragma omp for
	// 		#endif
	// 		for (uint64_t i=0; i<n_test; ++i) {
	// 			const auto psi = L_form<I>().test_dofs[i];
	// 			const uint64_t depth = psi.depth();
	// 			const QuadElem_t spt = q_map.s_el[depth];
	// 			const auto& X = q_map.p_qxa[depth];
	// 			const auto& Y = q_map.p_qya[depth];
	// 			const auto& Z = q_map.p_qza[depth];

	// 			L_form<I>().eval(vals, Jac[0], Jac[1], Jac[2],
	// 					psi, spt, X, Y, Z);

	// 			double val = 0.0;
	// 			#pragma omp simd reduction(+:val)
	// 			for (uint64_t l=0; l<NQ; ++l) {
	// 				val += vals[l] * q_map.p_qw[l];
	// 			}

	// 			L_form<I>().loc_val(i) = val;
	// 		}
	// 	}
	// }
}