#pragma once

#include<concepts>


namespace GV
{
	//when forwarding functions, we use the type std::nullptr_t as a compile-time flag.
	//this is a helpful check.
	template<typename T>
	concept NULLPTR_T = std::is_same_v<std::decay_t<T>, std::nullptr_t>;	
}
