#pragma once




namespace GV
{
	//a solver for symmetric elliptic problems of the form
	// Find u such that a(u,v)=b(v) for all v
	//where u and v come form the same test/trial space

	template<typename BilinearForm_type, typename LinearForm_type, typename GeometryPredicate_type>
	class SymmetricEllipticSolver
}