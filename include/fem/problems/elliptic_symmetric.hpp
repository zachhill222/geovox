#pragma once




namespace GV
{
	//a solver for symmetric elliptic problems of the form
	// Find u such that a(u,v)=b(v) for all v
	//where u and v come form the same test/trial space
	template<typename BilinearForm_type, typename LinearForm_type>
	class SymmetricEllipticSolver
	{
		//import type aliases
		using Mesh_t = typename BilinearForm_type::Mesh_t;
	}
}