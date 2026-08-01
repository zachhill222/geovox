#pragma once

#include "gutil.hpp"

#include "fem/dofhandler.hpp"


namespace GV {

	//////////////////////////////////////////////////////////////////////
	/// A helper class to track the current dofs and update coefficients
	/// after refinement.
	//////////////////////////////////////////////////////////////////////
	template<typename Handler_type>
	struct  DofCoefficients {
		

		//////////////////////////////////////////////////////////////////
		/// Aliases and constants
		//////////////////////////////////////////////////////////////////
		using Handler_t = Handler_type;
		using DOF_t = typename Handler_t::DOF_t;
		using Mesh_t = typename Handler_t::Mesh_t;
		using Elem_t = typename Handler_t::Elem_t;



	};





}


