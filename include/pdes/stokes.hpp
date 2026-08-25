#pragma once

#include "fem/fem.hpp"
#include "mesh/mesh.hpp"
#include "simd_keys/simd_keys.hpp"

namespace GV {
	

	/////////////////////////////////////////////////////////////
	/// A class to organize a diffuse stokes problem using
	/// Q1 dofs for pressure and Q1-iso-Q2 dofs for velocity.
	/// To implement the Q1-iso-Q2 dofs, we use the pressure dofs
	/// as the primary dofs and then uniformly refine them to get
	/// the velocity dofs.
	///
	/// Note that the velocity dofs are periodic while the pressure
	/// dofs are not.
	/////////////////////////////////////////////////////////////
	template<typename AssemblyType, int nQuadPoints=3>
	struct DiffuseStokes {
		

		/////////////////////////////////////////////////////////
		/// Aliases and constants
		/////////////////////////////////////////////////////////
		using Assembly_t    = AssemblyType;
		using MeshHandler_t = GV::DiffuseDomainMeshHandler<Assembly_t>;
		using Scalar_t      = typename Assembly_t::Scalar_t;
		using Point_t       = typename Assembly_t::Point_t;
		using Box_t 		= typename Assembly_t::Box_t;

		using Mesh_t      = typename MeshHandler_t::Mesh_t;
		using Elem_t      = typename Mesh_t::Elem_t;

		using P_DOF_t     = Keys::DOFS::VoxelQ1<0b000>;	//non-periodic
		using U_DOF_t     = Keys::DOFS::VoxelQ1<0b111>;	//fully-periodic

		using P_Handler_t = GV::DofHandler<Mesh_t, P_DOF_t>;
		using U_Handler_t = GV::DofHandler<Mesh_t, U_DOF_t>;

		using P_Coefs_t   = GV::CoefHandler<P_Handler_t, Scalar_t, 1>;
		using U_Coefs_t   = GV::CoefHandler<U_Handler_t, Scalar_t, 3>;

		using InteriorWeight_t = GV::AssemblyPhaseFieldWeight<true, Assembly_t>;
		using ExteriorWeight_t = GV::AssemblyPhaseFieldWeight<false, Assembly_t>;

		static constexpr int N_QUAD = nQuadPoints;


		/////////////////////////////////////////////////////////
		/// Constructors
		/////////////////////////////////////////////////////////
		explicit DiffuseStokes(const Box_t& domain, uint8_t max_depth) : 
			m_handler(domain, max_depth),
			p_handler(m_handler.mesh),
			u_handler(m_handler.mesh),
			p_coefs(p_handler),
			u_coefs(u_handler) {}


		/////////////////////////////////////////////////////////
		/// Problem data
		/////////////////////////////////////////////////////////
		Point_t          body_acceleration{1,0,0};
		Scalar_t         viscosity{1};


		/////////////////////////////////////////////////////////
		/// Primary data
		/////////////////////////////////////////////////////////
		MeshHandler_t m_handler;	//owns the mesh
		P_Handler_t   p_handler;
		U_Handler_t   u_handler;
		P_Coefs_t 	  p_coefs;
		U_Coefs_t     u_coefs;


		/////////////////////////////////////////////////////////
		/// Extra data for (bi)linear forms
		/////////////////////////////////////////////////////////
		InteriorWeight_t interior_wt{};	//phi
		ExteriorWeight_t exterior_wt{};	//1-phi
		Scalar_t		 eps_scale{1};	//eps = min_h * eps_scale


		/////////////////////////////////////////////////////////
		/// Iteration data
		/////////////////////////////////////////////////////////
		struct IterationData {
			Scalar_t abs_tol, rel_tol;
			int max_iterations;

			IterationData(Scalar_t atol, Scalar rtol, int miter) :
				abs_tol{atol}, rel_tol{rtol}, max_iterations{miter} {}
		};

		IterationData inner_iter{1e-6, 1e-6, 100};	//e.g., CG or GMRES
		IterationData outer_iter{1e-8, 1e-8, 50};	//e.g., Uzawa
		Scalar_t 	  relax_w{1};					//Uzawa relxation parameter


		/////////////////////////////////////////////////////////
		/// Utility methods
		/////////////////////////////////////////////////////////
		void initialize(uint8_t initial_depth) noexcept {
			mesh.set_depth(initial_depth);
			p_handler.init_dofs();

			set_velocity_dofs();
			p_coefs.init_coefs();	//set pressure to 0
			u_coefs.init_coefs();	//set each velocity component to 0
		}

		void set_velocity_dofs() {
			//set the velocity dofs to be the geometric children of the pressure dofs
			//this makes the velocity dofs Q1-iso-Q2 relative to the pressure so that
			//the inf-sup condition is satisfied. Additionally, refine the mesh so that
			//the quadrature happens on the finer velocity support elements.

			u_handler.clear();
			for (auto it=p_handler.active_dofs.begin(); it!=p_handler.active_dofs.end(); ++it) {
				U_DOF_t u_dof = static_cast<U_DOF_t>(*it);
				if (!u_dof.exists()) {continue;}

				for (U_DOF_t dof : u_dof.children()) {
					if (!dof.exists()) {continue;}
					u_handler.activate(dof);
				}
			}
			u_handler.collect_dofs();

			//the velocity dofs need finer quadrature elements
			m_handler.mesh.process_refine();
		}

		void geometry_refine(Scalar_t g_tol) {
			//refine pressure, update velocity
			Scalar_t tol_dist = g_tol * m_handler.min_element_size();
			std::vector<Elem_t> elems = m_handler.mesh.select_elements(
				[this,tol_dist](Elem_t el) {
					const auto pt = m_handler.mesh.geo_center(el);
					return gutil::abs(m_handler.assembly.signed_distance(pt)) < tol_dist;
				});

			p_handler.refine_quasi_hierarchical(elems);
			set_velocity_dofs();

			p_coefs.prolong_coefs();
			u_coefs.prolong_coefs();
		}
	};


}