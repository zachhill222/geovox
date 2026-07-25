#pragma once

//////////////////////////////////////////////////////
/// If OpenMP is present, some operations are provided with 
/// vectorized versions over spans. Note that if
/// a program is compiled agains OpenMP, then 
/// _OPENMP is defined.
//////////////////////////////////////////////////////


//////////////////////////////////////////////////////
/// Helper macros.
//////////////////////////////////////////////////////
#define GEOVOX_STRINGIFY(x) #x
#define GEOVOX_PRAGMA(x) _Pragma(#x)

#ifndef NDEBUG
	#define GEOVOX_DEBUG(...)	__VA_ARGS__ //inject in debug build
	#define GEOVOX_NDEBUG(...)				//supress in release build
#else
	#define GEOVOX_DEBUG(...) 				//supress in release build
	#define GEOVOX_NDEBUG(...) 	__VA_ARGS__ //inject in release build
#endif


//////////////////////////////////////////////////////
/// Macros to call OpenMP pragmas if compiled with openmp
//////////////////////////////////////////////////////
#ifdef _OPENMP
	#define GEOVOX_OMP(...) GEOVOX_PRAGMA(omp __VA_ARGS__)
	#define GEOVOX_SIMD(...) GEOVOX_PRAGMA(omp simd __VA_ARGS__)
	#define GEOVOX_DECLARE_SIMD(...) GEOVOX_PRAGMA(omp declare simd __VA_ARGS__)
#else
	#define GEOVOX_OMP(...)
	#define GEOVOX_SIMD(...)
	#define GEOVOX_DECLARE_SIMD(...)
#endif

