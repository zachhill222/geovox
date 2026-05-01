#pragma once

namespace GV
{
	//actions that linear and bilinear forms can take
	//for example, store interaction of basis functions to 
	//later construct sparse matrices or just compute the matrix-vector product
	//or the action of a linear form on some vector.

	//we may only wish to compute the action of a bilinear form against a specific vector (i.e., compute y=M*x without assembling M)
	//these types mark which action the bilinear form should take when scattering. these actions also provide some utility that is only
	//required with that action type (e.g., storing global values)

	//scatter the local matrix to the global. in a bilinear form, this will allow the sparse matrix M to be constructed
	//for requested test/trial basis functions. for a linear form, this will allow the vector to be build for requested test basis functions.
	struct ScatterAction {};
	template<typename T>
	concept ScatterActionType = std::same_as<T,ScatterAction>;

	//compute the action of a bilinear form on some input vector.
	//note the transpose is the same as the adjoint.
	//this action only makes sense for bilinear forms
	struct MatVecAction {}; //scatter the local matrix to compute M*x without assembling M
	template<typename T>
	concept MatVecActionType = std::same_as<T,MatVecAction> || std::same_as<T,MatVecAction>;

	//compute the action of a linear form on some input vector.
	struct DotAction {};
	template<typename T>
	concept DotActionType = std::same_as<T,DotAction>;
}