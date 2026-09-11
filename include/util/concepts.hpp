#pragma once

#include<concepts>


namespace GV
{
	//when forwarding functions, we use the type std::nullptr_t as a compile-time flag.
	//this is a helpful check.
	template<typename T>
	concept NULLPTR_T = std::is_same_v<std::decay_t<T>, std::nullptr_t>;

	template<typename T>
	concept VOID_T = std::is_same_v<std::decay_t<T>, void>;

	//Check if all types of a variadic argument are the same
	template<typename Target, typename... Args>
	concept AllArgsSameAs = (std::same_as<Target,Args> && ...);
}
